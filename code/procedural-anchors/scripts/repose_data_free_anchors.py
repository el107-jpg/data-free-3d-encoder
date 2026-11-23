"""Put an existing anchor file into another pose rule (``pose_power``) without the source meshes.

A shape moved by ``c`` and resized by ``s`` has the field ``f'(x) = s f(x / s + c)``, so the stored distance
grids can be re-posed by resampling. The centre and size are measured as the trainer measures them
(soft inside indicator, power ``pose_power`` of the distance from the origin); orientation is left as it is
(the principal axes of the mass do not change under a move or a resize). Two rounds, composed, then one
resampling from the original grid. Outside the stored cube the field is continued by the distance to the cube.

usage: repose_data_free_anchors.py <anchors.pt> <new folder> [--pose-power 6]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import scipy.ndimage as ndi
import torch

from arm_planner.data_free import losses as L
from arm_planner.data_free.config import DataFreeConfig


def sample(grid: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Field at ``points`` (N x 3 in [-1, 1]^3 coordinates, or beyond)."""
    res = grid.shape[0]
    inside = np.clip(points, -1.0, 1.0)
    beyond = np.linalg.norm(points - inside, axis=1)
    values = ndi.map_coordinates(grid, ((inside + 1.0) / 2.0 * (res - 1)).T, order=1, mode="nearest")
    return values - beyond


def extent(t: np.ndarray, points: np.ndarray, tau: float, power: float):
    w = 1.0 / (1.0 + np.exp(-np.clip(t / tau, -60, 60)))
    centre = (w[:, None] * points).sum(0) / (w.sum() + 1e-6)
    for _ in range(8 if power != 2.0 else 0):
        y = points - centre
        rho = np.maximum(np.linalg.norm(y, axis=1), 1e-9)
        q = w * rho ** (power - 2.0)
        unit = y / rho[:, None]
        hessian = q.sum() * np.eye(3) + (power - 2.0) * np.einsum("n,ni,nj->ij", q, unit, unit)
        centre = centre + np.linalg.solve(hessian, (q[:, None] * y).sum(0))
    rho = np.linalg.norm(points - centre, axis=1)
    return centre, float(((w * rho ** power).sum() / (w.sum() + 1e-6)) ** (1.0 / power))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("output_dir")
    ap.add_argument("--pose-power", type=float, default=None)
    args = ap.parse_args()
    if os.path.exists(os.path.join(args.output_dir, "anchors.pt")):
        raise SystemExit(f"{args.output_dir} already holds anchors.pt; refusing to overwrite")
    data = torch.load(args.source, map_location="cpu", weights_only=False)
    power = float(DataFreeConfig().pose_power if args.pose_power is None else args.pose_power)
    cfg = DataFreeConfig(pose_power=power)
    tau, target = float(cfg.soft_tau), L.pose_size_target(cfg)
    grids = data["grid"].float().numpy()
    if not bool(data.get("positive_inside", True)):
        grids = -grids
    res = grids.shape[1]
    axis = np.linspace(-1.0, 1.0, res)
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
    radius = np.linalg.norm(points, axis=1)
    out, records = [], []
    for k, grid in enumerate(grids):
        shift, scale = np.zeros(3), 1.0
        for _ in range(3):
            t = scale * sample(grid, points / scale + shift)
            centre, size = extent(t, points, tau, power)
            step = target / size
            shift, scale = shift + centre / scale, scale * step       # f'(x) = S f(x / S + C)
        t = scale * sample(grid, points / scale + shift)
        centre, size = extent(t, points, tau, power)
        solid = t > 0
        records.append({"index": k, "file_id": int(data["file_ids"][k]), "resize": float(scale),
                        "move": float(np.linalg.norm(shift)), "size": size, "centre_offset": float(np.linalg.norm(centre)),
                        "reach": float(radius[solid].max()) if solid.any() else 0.0,
                        "volume_fraction": float(solid.mean())})
        out.append(t.reshape(res, res, res).astype(np.float16))
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save({"grid": torch.from_numpy(np.stack(out)), "resolution": res, "positive_inside": True,
                "pose_power": power, "file_ids": list(data["file_ids"])}, os.path.join(args.output_dir, "anchors.pt"))
    json.dump({"source": args.source, "pose_power": power, "size_target": target, "anchors": records},
              open(os.path.join(args.output_dir, "anchors.json"), "w"), indent=1)
    reach = np.array([r["reach"] for r in records]); resize = np.array([r["resize"] for r in records])
    print(f"{len(records)} anchors re-posed for pose_power {power:g}: resize {resize.min():.2f} to {resize.max():.2f} "
          f"(median {np.median(resize):.2f}); size error at most {max(abs(r['size'] / target - 1) for r in records):.4f}; "
          f"centre offset at most {max(r['centre_offset'] for r in records):.4f}; reach {reach.min():.2f} to {reach.max():.2f}, "
          f"{int((reach > 0.83).sum())} beyond 0.83")
    print("saved", os.path.join(args.output_dir, "anchors.pt"))


if __name__ == "__main__":
    main()
