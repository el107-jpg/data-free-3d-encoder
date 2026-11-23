#!/usr/bin/env python3
"""Pick anchor objects for the data-free latent from Thingi10K and store their distance grids.

An anchor is a real object the decoder is asked to reproduce at a fixed code. To be reproducible
by the decoder at all, an object has to respect the decoder's hard limits and the trainer's pose
rules, so every candidate is

1. taken from the prepared Thingi10K originals whose metadata says: one piece, no handles
   (Euler characteristic 2), closed, manifold, oriented, no self-intersections;
2. pose-normalised the way the trainer measures pose: soft centroid at the origin, soft
   covariance diagonal (axis order and signs drawn at random, the trainer does not pin them),
   soft covariance trace equal to ``cfg.cov_trace_target``;
3. kept only if, at that size, it contains the ball of radius ``inner`` around the origin, stays
   inside radius ``outer``, is one piece without cavities or handles on the grid, and loses
   little volume when features thinner than ``cfg.min_feature`` are removed (and the same for
   thin gaps).

From the eligible objects, ``--count`` are chosen by farthest-point selection on the distance
the trainer itself uses between shapes (mean absolute field difference inside the bound radius),
so the anchors are as unlike each other as the pool allows.

Outputs (``--output-dir``): ``anchors.pt`` (``grid``: K x R x R x R float16, field positive
inside, ``grid[k, i, j, l]`` at ``(x_i, y_j, z_l)`` on ``linspace(-1, 1, R)``), ``anchors.json``
(sources, licences, transforms, statistics, selection order) and a contact sheet.

Signed distances come from the project's Warp oracle (ray-parity sign, float64 refinement).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import scipy.ndimage as ndi
import torch
import trimesh
from scipy.optimize import linear_sum_assignment

from arm_planner.data_free import diagnostics as D
from arm_planner.data_free.config import DataFreeConfig
from arm_planner.object_geometry.warp_oracle import WarpGeometryOracle


def metadata_ok(meta: dict) -> bool:
    g = meta.get("source_geometry_metadata", {})

    def val(key, default="nan"):
        try:
            return float(g.get(key, default))
        except (TypeError, ValueError):
            return float("nan")

    return (val("num_connected_components") == 1 and val("euler_characteristic") == 2
            and val("num_boundary_edges") == 0 and val("vertex_manifold") == 1 and val("edge_manifold") == 1
            and val("oriented") == 1 and val("num_self_intersections") == 0 and val("solid", "1") == 1)


def grid_points(resolution: int) -> np.ndarray:
    axis = np.linspace(-1.0, 1.0, resolution)
    return np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3).astype(np.float32)


def near_identity_rotation(cov: np.ndarray) -> np.ndarray:
    """Rotation (rows = new axes) that diagonalises ``cov`` and moves the axes as little as possible."""
    _, vec = np.linalg.eigh(cov)
    rows, cols = linear_sum_assignment(-np.abs(vec))                # axis <- eigenvector
    rot = np.zeros((3, 3))
    for axis, col in zip(rows, cols):
        v = vec[:, col]
        rot[axis] = v if v[axis] >= 0 else -v
    if np.linalg.det(rot) < 0:
        rot[np.argmin(np.abs(np.diag(rot)))] *= -1
    return rot


class Fields:
    """Positive-inside field of a mesh at fixed grid points, with the trainer's soft pose statistics."""

    def __init__(self, cfg, resolution: int, device: str, sign: float) -> None:
        self.cfg, self.resolution, self.device, self.sign = cfg, resolution, device, sign
        self.points = grid_points(resolution)
        self.radius = np.linalg.norm(self.points, axis=1)
        self.spacing = 2.0 / (resolution - 1)

    def field(self, vertices: np.ndarray, faces: np.ndarray, points: np.ndarray = None) -> np.ndarray:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        oracle = WarpGeometryOracle.from_mesh(mesh, family="solid", device=self.device)
        return self.sign * oracle(self.points if points is None else points)

    def moments(self, t: np.ndarray):
        w = 1.0 / (1.0 + np.exp(-np.clip(t / float(self.cfg.soft_tau), -60, 60)))
        total = w.sum() + 1e-6
        centroid = (w[:, None] * self.points).sum(0) / total
        off = self.points - centroid
        cov = (w[:, None] * off).T @ off / total
        return centroid, cov


def normalise(vertices, faces, fields: Fields, rng, inner: float, rounds: int = 3):
    """Vertices in the trainer's pose convention, or None with a reason."""
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if not mesh.is_watertight or not mesh.is_winding_consistent:
        return None, "not watertight"
    volume = float(mesh.volume)
    if not np.isfinite(volume) or abs(volume) < 1e-12:
        return None, "no volume"
    if volume < 0:
        faces = faces[:, ::-1].copy()
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        volume = -volume
    inertia = np.asarray(mesh.moment_inertia, dtype=np.float64)
    cov = (np.trace(inertia) / 2.0 * np.eye(3) - inertia) / volume
    lam, vec = np.linalg.eigh(cov)
    if lam.min() <= 0 or not np.isfinite(lam).all():
        return None, "bad inertia"
    perm = rng.permutation(3)
    rot = vec[:, perm].T * rng.choice([-1.0, 1.0], size=3)[:, None]   # random axis order and signs
    if np.linalg.det(rot) < 0:
        rot[0] *= -1
    target = float(fields.cfg.cov_trace_target)
    v = (vertices - np.asarray(mesh.center_mass)) @ rot.T * np.sqrt(target / np.trace(cov))
    if np.linalg.norm(v, axis=1).max() > 0.95:
        return None, "too long for the outer radius"
    # cheap early exit (one query point) before the grid rounds; the exact test comes later
    if float(fields.field(v, faces, np.zeros((1, 3), dtype=np.float32))[0]) < inner - 0.05:
        return None, "centre too shallow"
    centroid, cov = None, None
    for _ in range(rounds):                                           # match the trainer's soft statistics
        t = fields.field(v, faces)
        centroid, cov = fields.moments(t)
        v = (v - centroid) @ near_identity_rotation(cov).T * np.sqrt(target / np.trace(cov))
    return (v, faces), ""


def eligibility(v, faces, fields: Fields, inner: float, outer: float, thin_limit: float):
    cfg, res = fields.cfg, fields.resolution
    if np.linalg.norm(v, axis=1).max() > outer:
        return None, "outside the outer radius"
    depth = float(fields.field(v, faces, np.zeros((1, 3), dtype=np.float32))[0])
    if depth < inner:
        return None, "centre too shallow"
    t = fields.field(v, faces)
    centroid, cov = fields.moments(t)
    grid = t.reshape(res, res, res)
    solid = grid > 0
    topo = D._topology_of_mask(solid)
    if topo["pieces"] != 1 or topo["cavities"] != 0 or topo["handles"] != 0:
        return None, "not one simple piece on the grid"
    half, h = 0.5 * float(cfg.min_feature), fields.spacing
    core = grid > half
    if not core.any():
        return None, "thinner than the minimum feature everywhere"
    opened = ndi.distance_transform_edt(~core, sampling=h) <= half + 0.5 * h
    thin = float((solid & ~opened).sum()) / float(solid.sum())
    ball = fields.radius.reshape(res, res, res) < float(cfg.bound_radius)
    far = grid < -half
    closed = ndi.distance_transform_edt(~far, sampling=h) <= half + 0.5 * h
    gap = float((~solid & ball & ~closed).sum()) / float(solid.sum())
    if thin > thin_limit:
        return None, "thin features"
    if gap > thin_limit:
        return None, "thin gaps"
    off = cov - np.diag(np.diag(cov))
    stats = {"depth_at_centre": depth, "max_radius": float(np.linalg.norm(v, axis=1).max()),
             "thin_fraction": thin, "gap_fraction": gap, "volume_fraction": float(solid.mean()),
             "centroid_offset": float(np.linalg.norm(centroid)), "trace": float(np.trace(cov)),
             "offdiag_max": float(np.abs(off).max()),
             "axes": [float(x) for x in np.sqrt(np.sort(np.diag(cov))[::-1])]}
    return (t.astype(np.float16), stats), ""


def contact_sheet(grids: np.ndarray, titles, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage import measure

    count = grids.shape[0]
    cols = 8
    rows = int(np.ceil(count / cols))
    fig = plt.figure(figsize=(2.1 * cols, 2.2 * rows))
    res = grids.shape[1]
    for k in range(count):
        ax = fig.add_subplot(rows, cols, k + 1, projection="3d")
        verts, tris, _, _ = measure.marching_cubes(grids[k].astype(np.float32), level=0.0, step_size=2)
        verts = verts / (res - 1) * 2.0 - 1.0
        ax.plot_trisurf(verts[:, 0], verts[:, 1], tris, verts[:, 2], color="#7fa8d6", edgecolor="none",
                        linewidth=0, antialiased=False, shade=True)
        ax.set_xlim(-0.6, 0.6); ax.set_ylim(-0.6, 0.6); ax.set_zlim(-0.6, 0.6)
        ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=22, azim=35); ax.set_axis_off()
        ax.set_title(titles[k], fontsize=7, pad=-4)
    fig.suptitle(f"{count} anchor objects (Thingi10K), in selection order: each is the one least like those before it",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    root = "/home/eric-luo/Downloads/pointcloud-motion-planning/datasets"
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=f"{root}/thingi10k_source_v1_5_0/prepared_v2")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--count", type=int, default=64)
    ap.add_argument("--resolution", type=int, default=96)
    ap.add_argument("--check-resolution", type=int, default=64)
    ap.add_argument("--inner", type=float, default=0.17, help="depth needed at the centre (decoder limit 0.15)")
    ap.add_argument("--outer", type=float, default=0.83, help="largest radius allowed (decoder limit 0.85)")
    ap.add_argument("--thin-limit", type=float, default=0.06)
    ap.add_argument("--min-faces", type=int, default=12)
    ap.add_argument("--max-candidates", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sheet", default="")
    ap.add_argument("--idle", type=float, default=0.3, help="idle time per object as a fraction of its busy time")
    ap.add_argument("--save-eligible", action="store_true",
                    help="also store the grids of every eligible object (eligible.pt), with a flag for the chosen ones")
    ap.add_argument("--cov-trace-target", type=float, default=None,
                    help="size the objects are normalised to (default: the trainer's default target)")
    ap.add_argument("--base-radius", type=float, default=None,
                    help="radius of the ball the selection starts farthest from (default: the trainer's default)")
    args = ap.parse_args()
    if os.path.exists(os.path.join(args.output_dir, "anchors.pt")):
        raise SystemExit(f"{args.output_dir} already holds anchors.pt; refusing to overwrite")
    os.makedirs(args.output_dir, exist_ok=True)
    cfg = DataFreeConfig()
    if args.cov_trace_target is not None:
        cfg.cov_trace_target = float(args.cov_trace_target)
    if args.base_radius is not None:
        cfg.base_radius = float(args.base_radius)

    # the oracle's sign convention, measured on a ball
    ball = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
    probe = WarpGeometryOracle.from_mesh(ball, family="solid", device=args.device)(np.zeros((1, 3), dtype=np.float32))[0]
    sign = 1.0 if probe > 0 else -1.0
    check = Fields(cfg, args.check_resolution, args.device, sign)
    final = Fields(cfg, args.resolution, args.device, sign)

    paths = sorted(glob.glob(os.path.join(args.source, "*.json")))
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(paths))
    reasons, eligible = {}, []
    start, seen = time.time(), 0
    for index in order:
        try:
            meta = json.load(open(paths[index]))
        except (UnicodeDecodeError, json.JSONDecodeError):               # a damaged file is counted, not fatal
            reasons["unreadable metadata"] = reasons.get("unreadable metadata", 0) + 1
            continue
        if not metadata_ok(meta):
            reasons["metadata"] = reasons.get("metadata", 0) + 1
            continue
        mesh_path = paths[index][:-5] + ".npz"
        if not os.path.exists(mesh_path):
            reasons["no mesh file"] = reasons.get("no mesh file", 0) + 1
            continue
        if args.max_candidates and seen >= args.max_candidates:
            break
        seen += 1
        t0 = time.time()
        data = np.load(mesh_path)
        vertices, faces = np.asarray(data["vertices"], dtype=np.float64), np.asarray(data["faces"], dtype=np.int64)
        reason = ""
        if len(faces) < args.min_faces:
            reason = "too few faces"
        else:
            try:
                shape, reason = normalise(vertices, faces, check,
                                          np.random.default_rng([args.seed, int(meta["file_id"])]), args.inner)
                if shape is not None:
                    result, reason = eligibility(shape[0], shape[1], check, args.inner, args.outer, args.thin_limit)
                    if result is not None:
                        eligible.append({"meta": meta, "v": shape[0], "f": shape[1], "t": result[0], "stats": result[1]})
            except Exception as error:                                    # a bad source is skipped, and counted
                reason = f"error: {type(error).__name__}"
        if reason:
            reasons[reason] = reasons.get(reason, 0) + 1
        time.sleep(args.idle * (time.time() - t0))
        if seen % 100 == 0:
            print(f"{seen} candidates, {len(eligible)} eligible, {time.time() - start:.0f} s", flush=True)
    print(f"{seen} candidates with good metadata; {len(eligible)} eligible; rejected: {reasons}", flush=True)
    if len(eligible) < args.count:
        raise SystemExit(f"only {len(eligible)} eligible objects for {args.count} anchors")

    # farthest-point selection on the trainer's shape distance
    inside = torch.from_numpy(check.radius < float(cfg.bound_radius))
    stack = torch.from_numpy(np.stack([e["t"] for e in eligible]))[:, inside].float()
    sphere = torch.from_numpy(float(cfg.base_radius) - check.radius)[inside].float()
    nearest = (stack - sphere).abs().mean(dim=1)                      # start: least like the ball
    chosen, gaps = [], []
    for _ in range(args.count):
        pick = int(torch.argmax(nearest))
        chosen.append(pick)
        gaps.append(float(nearest[pick]))
        distance = (stack - stack[pick]).abs().mean(dim=1)
        nearest = distance if len(chosen) == 1 else torch.minimum(nearest, distance)
        nearest[chosen] = -1.0
    pair = torch.stack([(stack[chosen] - stack[c]).abs().mean(dim=1) for c in chosen])
    off = pair[~torch.eye(len(chosen), dtype=torch.bool)]
    print(f"chosen {len(chosen)}: pairwise distance min {float(off.min()):.4f} median {float(off.median()):.4f} "
          f"max {float(off.max()):.4f}; distance to the ball median {float((stack[chosen] - sphere).abs().mean(1).median()):.4f}",
          flush=True)

    grids, records = [], []
    for rank, pick in enumerate(chosen):
        e = eligible[pick]
        grid = final.field(e["v"], e["f"]).reshape(args.resolution, args.resolution, args.resolution)
        grids.append(grid.astype(np.float16))
        m = e["meta"]
        records.append({"rank": rank, "file_id": int(m["file_id"]), "thing_id": m.get("thing_id"), "name": m.get("name"),
                        "author": m.get("author"), "license": m.get("license"), "category": m.get("category"),
                        "faces": int(len(e["f"])), "selection_gap": gaps[rank], **e["stats"]})
    grids = np.stack(grids)
    torch.save({"grid": torch.from_numpy(grids), "resolution": args.resolution, "positive_inside": True,
                "file_ids": [r["file_id"] for r in records]}, os.path.join(args.output_dir, "anchors.pt"))
    json.dump({"source": args.source, "arguments": vars(args), "candidates": seen, "eligible": len(eligible),
               "rejected": reasons, "cov_trace_target": float(cfg.cov_trace_target), "soft_tau": float(cfg.soft_tau),
               "pairwise_distance": {"min": float(off.min()), "median": float(off.median()), "max": float(off.max())},
               "anchors": records}, open(os.path.join(args.output_dir, "anchors.json"), "w"), indent=1)
    print("saved", os.path.join(args.output_dir, "anchors.pt"), flush=True)
    if args.save_eligible:
        # every eligible object at the final resolution, chosen or not (for held-out fit tests)
        picked = set(chosen)
        every = np.stack([final.field(e["v"], e["f"]).reshape(args.resolution, args.resolution, args.resolution)
                          .astype(np.float16) for e in eligible])
        torch.save({"grid": torch.from_numpy(every), "resolution": args.resolution, "positive_inside": True,
                    "file_ids": [int(e["meta"]["file_id"]) for e in eligible],
                    "names": [str(e["meta"].get("name")) for e in eligible],
                    "chosen": [i in picked for i in range(len(eligible))]},
                   os.path.join(args.output_dir, "eligible.pt"))
        print("saved", os.path.join(args.output_dir, "eligible.pt"), flush=True)
    if args.sheet:
        contact_sheet(grids, [f'{r["rank"]}: {str(r["name"])[:22]}' for r in records], args.sheet)
        print("saved", args.sheet, flush=True)


if __name__ == "__main__":
    main()
