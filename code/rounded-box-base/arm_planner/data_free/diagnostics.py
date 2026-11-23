"""Diagnostics for the data-free latent space (logged, never a loss).

Three independent checks:

* ``grid_topology`` / ``decoder_grid_topology``: pieces, cavities and handles of
  a field on a voxel grid (plan section 8, "count handles on anchor codes").
* ``graph_topology`` / ``compare_graph_to_grid``: the plan's "check before using
  it in a loss": do the sample graph and the grid agree on pieces and cavities?
* ``dead_directions`` / ``subspace_stability`` / ``summarise``: plan section 7.

Sign convention: positive inside. The spatial gradient points into the solid,
so the outward unit normal is ``-g / |g|``.

``scipy`` and ``scikit-image`` are imported lazily, only by ``grid_topology``.
"""

from __future__ import annotations

import math
from typing import Callable, Mapping, Optional, Sequence, Union

import numpy as np
import torch
from torch import Tensor

__all__ = [
    "grid_topology",
    "decoder_grid_topology",
    "graph_topology",
    "graph_vs_grid",
    "compare_graph_to_grid",
    "dead_directions",
    "subspace_stability",
    "summarise",
]


# ---------------------------------------------------------------------------
# Grid topology
# ---------------------------------------------------------------------------


def _grid_points(resolution: int, box: float, device) -> Tensor:
    """Cell centres of a ``resolution^3`` grid over ``[-box, box]^3``, M×3."""
    h = 2.0 * box / resolution
    axis = torch.linspace(-box + h / 2, box - h / 2, resolution, device=device)
    gx, gy, gz = torch.meshgrid(axis, axis, axis, indexing="ij")
    return torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)


def _topology_of_mask(solid: np.ndarray) -> dict:
    """Pieces, cavities, Euler number and handles of a boolean voxel solid.

    Connectivity pairing: the solid is 26-connected (voxels touching by a face,
    edge or corner are joined) and the free space is 6-connected (faces only).
    This is the pairing ``skimage.measure.euler_number(..., connectivity=3)``
    assumes, so ``euler = pieces - handles + cavities`` holds exactly.
    """
    from scipy import ndimage
    from skimage.measure import euler_number

    # One layer of free space all round: every free region that reaches the box
    # boundary joins a single exterior component, so
    # cavities = free components - 1.
    padded = np.pad(np.asarray(solid, dtype=bool), 1, constant_values=False)
    _, pieces = ndimage.label(padded, structure=np.ones((3, 3, 3), dtype=bool))
    _, free = ndimage.label(~padded, structure=ndimage.generate_binary_structure(3, 1))
    cavities = int(free) - 1
    euler = int(euler_number(padded, connectivity=3))
    return {
        "pieces": int(pieces),
        "cavities": cavities,
        "euler": euler,
        "handles": int(pieces) + cavities - euler,
    }


@torch.no_grad()
def grid_topology(
    field_fn: Callable[[Tensor], Tensor],
    resolution: int = 64,
    box: float = 1.0,
    device=None,
    chunk: int = 65536,
) -> dict:
    """Topology counts of ``{f > 0}`` on a voxel grid over ``[-box, box]^3``.

    ``field_fn(points M×3) -> values M`` (positive inside), evaluated at cell
    centres in chunks of ``chunk`` points.

    Returns ``pieces`` (components of ``f > 0``), ``cavities`` (components of
    ``f <= 0`` that do not touch the box boundary), ``euler`` and
    ``handles = pieces + cavities - euler`` (Euler characteristic of a 3-D solid
    is ``b0 - b1 + b2``), plus ``volume_fraction`` and ``resolution``.

    Connectivity: 26 for the solid, 6 for free space (see ``_topology_of_mask``).
    Features thinner than one cell (``2 * box / resolution``) are not resolved.
    """
    points = _grid_points(resolution, box, device)
    values = []
    for start in range(0, points.shape[0], chunk):
        values.append(field_fn(points[start : start + chunk]).reshape(-1).float().cpu())
    solid = (torch.cat(values) > 0).reshape(resolution, resolution, resolution).numpy()
    out = _topology_of_mask(solid)
    out["volume_fraction"] = float(solid.mean())
    out["resolution"] = int(resolution)
    return out


@torch.no_grad()
def decoder_grid_topology(
    decoder,
    codes: Tensor,
    cfg=None,
    resolution: int = 64,
    box: float = 1.0,
    chunk: int = 65536,
) -> list:
    """``grid_topology`` for each code (B×D) of a decoder; a list of B dicts.

    The decoder needs ``prepare(codes) -> mod`` and ``query(mod, x B×N×3) -> B×N``.
    Codes are handled one at a time so memory stays at ``chunk`` points.
    ``cfg`` is accepted for a uniform call signature; it is not needed.
    """
    results = []
    for i in range(codes.shape[0]):
        mod = decoder.prepare(codes[i : i + 1])
        results.append(
            grid_topology(
                lambda p: decoder.query(mod, p[None])[0],
                resolution=resolution,
                box=box,
                device=codes.device,
                chunk=chunk,
            )
        )
    return results


# ---------------------------------------------------------------------------
# Graph versus grid
# ---------------------------------------------------------------------------


def graph_topology(graph, from_rows: bool = False) -> dict:
    """Per-shape piece and cavity counts reported by a ``ShapeGraph``.

    By default the graph's own uncapped counts are used when it carries them
    (``num_pieces``: optimistic solid components with a confident inside node,
    main body included, so 0 for an empty shape; ``num_cavities``: optimistic
    outside components with no far-field node). These are the counts at build
    time.

    With ``from_rows=True``, or for a graph without those fields,
    ``pieces = 1 + number of kind-0 fragment rows`` and ``cavities = number of
    cavity rows``. Rows are capped by ``max_fragments`` / ``max_cavities`` and
    reflect whatever survives in the graph passed in, so use this on a graph
    returned by ``confirm`` to count confirmed findings only.

    Returns long tensors of length B on the graph's device.
    """
    if not from_rows and getattr(graph, "num_pieces", None) is not None \
            and getattr(graph, "num_cavities", None) is not None:
        return {"pieces": graph.num_pieces.long(), "cavities": graph.num_cavities.long()}
    batch = int(graph.main_label.shape[0])
    kind0 = graph.frag_kind.reshape(-1) == 0
    frag_shape = graph.frag_shape.reshape(-1).long()[kind0]
    pieces = 1 + torch.bincount(frag_shape, minlength=batch)
    cavities = torch.bincount(graph.cav_shape.reshape(-1).long(), minlength=batch)
    return {"pieces": pieces, "cavities": cavities}


def graph_vs_grid(
    decoder,
    codes: Tensor,
    cfg,
    resolution: int = 48,
    generator: Optional[torch.Generator] = None,
    mode: str = "full",
    from_rows: bool = False,
    confirmed: bool = True,
) -> dict:
    """The plan's "check before using it in a loss", on a batch of codes.

    Evaluates the field once per code at ``cfg.points`` fresh samples, builds
    the sample graph (``mode`` as in ``build_graph``, consistency error = the
    batch median ``|f(xp)|``), counts pieces and cavities on a voxel grid for
    the same codes and returns ``compare_graph_to_grid`` plus the two sets of
    counts (``graph_counts``, ``grid_counts``) and ``consistency_error``.

    ``confirm`` is part of the method, so by default (``confirmed=True``) the
    graph's findings are re-checked with the decoder first and the counts are
    those after it (``confirm`` keeps ``num_pieces`` / ``num_cavities`` up to
    date). ``confirm_dropped`` reports what it removed (kind-0 fragments,
    kind-1 fragments, cavities) and ``graph_counts_raw`` the counts before.
    Needs ``field.py``, ``graph.py`` and ``graph_losses.py``.
    """
    from .field import evaluate_field, sample_points
    from .graph import build_graph
    from .graph_losses import confirm

    device = codes.device
    x = sample_points(cfg, 1, generator, device)
    cluster = torch.zeros(codes.shape[0], dtype=torch.long, device=device)
    fb = evaluate_field(decoder, codes, cluster, x, cfg, create_graph=False)
    e = float(fb.fp.detach().abs().median())
    with torch.no_grad():
        graph = build_graph(fb, e, cfg, mode=mode)
        raw_counts = graph_topology(graph, from_rows=from_rows)
        if confirmed:
            graph = confirm(decoder, fb.mod, graph, cfg)
    graph_counts = graph_topology(graph, from_rows=from_rows)
    grid_counts = decoder_grid_topology(decoder, codes, cfg, resolution=resolution)
    out = compare_graph_to_grid(graph_counts, grid_counts)
    out["graph_counts"] = {k: v.cpu() for k, v in graph_counts.items()}
    out["graph_counts_raw"] = {k: v.cpu() for k, v in raw_counts.items()}
    out["confirm_dropped"] = {
        key: int(graph.stats.get(f"confirm_dropped_{key}", 0))
        for key in ("kind0", "kind1", "cavities")
    }
    out["grid_counts"] = grid_counts
    out["consistency_error"] = e
    return out


def _count_tensor(counts, key: str) -> Tensor:
    if isinstance(counts, Mapping):
        return torch.as_tensor(counts[key]).detach().cpu().long().reshape(-1)
    return torch.tensor([int(c[key]) for c in counts], dtype=torch.long)


def compare_graph_to_grid(
    graph_counts: Mapping,
    grid_counts: Union[Mapping, Sequence[Mapping]],
) -> dict:
    """Agreement between graph and grid counts over a batch of shapes.

    ``graph_counts``: the dict from ``graph_topology``. ``grid_counts``: either
    a dict of per-shape tensors with the same keys or the list of dicts from
    ``decoder_grid_topology``.

    Returns, as floats, the fraction of shapes on which the two agree on
    ``pieces``, on ``cavities`` and on both, the fractions where the graph
    reports more / fewer than the grid, and ``count`` (number of shapes).
    The plan asks for disagreement of at most a few percent.
    """
    out: dict = {}
    both = None
    for key in ("pieces", "cavities"):
        a = _count_tensor(graph_counts, key)
        b = _count_tensor(grid_counts, key)
        if a.shape != b.shape:
            raise ValueError(f"{key}: graph has {a.numel()} shapes, grid has {b.numel()}")
        same = a == b
        both = same if both is None else both & same
        n = max(int(a.numel()), 1)
        out[f"{key}_agreement"] = float(same.sum()) / n
        out[f"{key}_graph_more"] = float((a > b).sum()) / n
        out[f"{key}_graph_fewer"] = float((a < b).sum()) / n
        out["count"] = int(a.numel())
    out["both_agreement"] = float(both.sum()) / max(out["count"], 1)
    return out


# ---------------------------------------------------------------------------
# Dead directions
# ---------------------------------------------------------------------------


def _tangent_basis(z: Tensor) -> Tensor:
    """(D-1)×D orthonormal basis of the tangent space at unit code ``z``."""
    try:
        from .latent import tangent_basis

        return tangent_basis(z)
    except ImportError:  # latent.py not present: same result up to rotation
        d = z.shape[-1]
        zz = (z / z.norm()).double().reshape(d, 1)
        q, _ = torch.linalg.qr(torch.cat([zz, torch.eye(d, dtype=torch.float64, device=z.device)], dim=1))
        return q[:, 1:d].T.to(z.dtype)


def _field_and_grad(decoder, mod, x: Tensor):
    """Field (N) and spatial gradient (N×3) for one code at points ``x`` (N×3)."""
    with torch.enable_grad():
        xr = x.detach().clone().requires_grad_(True)
        f = decoder.query(mod, xr[None])[0]
        (g,) = torch.autograd.grad(f.sum(), xr)
    return f.detach(), g.detach()


def _neighbour_mean(points: Tensor, values: Tensor, radius: float, chunk: int = 1024) -> Tensor:
    """Replace each row of ``values`` by the mean over points within ``radius``."""
    out = torch.empty_like(values)
    for start in range(0, points.shape[0], chunk):
        near = (torch.cdist(points[start : start + chunk], points) <= radius).to(values.dtype)
        out[start : start + chunk] = (near @ values) / near.sum(dim=1, keepdim=True)
    return out


def _empty_result(dim: int, e: float, threshold: float, surface_points: int) -> dict:
    return {
        "singular_values": torch.zeros(dim - 1),
        "singular_values_unsmoothed": torch.zeros(dim - 1),
        "directions": torch.zeros(dim - 1, dim),
        "live": 0,
        "pose_share": 0.0,
        "wobble_share": 0.0,
        "consistency_error": e,
        "threshold": threshold,
        "rms_motion": 0.0,
        "surface_points": surface_points,
    }


def dead_directions(
    decoder,
    code: Tensor,
    cfg=None,
    points: int = 4096,
    planner_resolution: float = 0.05,
    cluster_radius: float = 0.2,
    step: float = 1e-3,
    generator: Optional[torch.Generator] = None,
    surface_factor: float = 3.0,
    surface_floor: float = 1e-4,
    min_motion: float = 1e-4,
    code_chunk: int = 16,
) -> dict:
    """Plan section 7: how many latent directions really move the surface.

    Steps, for one unit ``code`` (D):

    1. Surface points. ``points`` uniform samples in the box, one-step guesses
       ``xp = x - f*g``; the consistency error ``e`` is the median ``|f(xp)|``.
       Guesses with ``|f(xp)| <= max(surface_factor * e, surface_floor)`` are
       the surface points ``p``; the gradient is re-evaluated there and the
       outward normal is ``n = -g/|g|``.
    2. For each of the D-1 tangent directions ``u_k`` at the code, the field
       derivative at ``p`` by central difference in the code,
       ``(f(norm(z + step*u_k)) - f(norm(z - step*u_k))) / (2*step)``, turned
       into outward surface displacement per unit chord, ``d_k = df_k / |g|``.
    3. Pose removal: least-squares projection of every displacement field off
       the span of ``n·e_i`` (translation), ``n·(e_i × p)`` (rotation) and
       ``n·p`` (uniform scale). ``pose_share`` = energy removed / total energy.
    4. Smoothing: each point's displacement becomes the mean over surface
       points within ``planner_resolution``. ``wobble_share`` = 1 - energy
       after / energy before.
    5. SVD of the smoothed (points × (D-1)) matrix over sqrt(points). Singular
       values are RMS surface displacement per unit chord. A direction is live
       when ``s * cluster_radius > max(e, min_motion)``.

    ``min_motion`` is a floor on the liveness threshold: for a nearly exact
    field ``e`` drops to float rounding, below the noise of the finite
    difference itself. Set it to 0 for the plan's bare criterion. The same
    floor guards ``wobble_share``: it is 0 when even the strongest direction
    left after pose removal moves less than ``min_motion`` over a cluster.

    Returns a dict (tensors on the CPU):
      ``singular_values`` (D-1, descending), ``singular_values_unsmoothed``
      (after pose removal, before smoothing), ``directions`` ((D-1)×D right
      singular vectors as unit tangent vectors in code space, row i matching
      singular value i), ``live`` (int), ``pose_share``, ``wobble_share``,
      ``consistency_error``, ``threshold``, ``rms_motion`` (RMS displacement
      per unit chord over all directions before pose removal) and
      ``surface_points`` (int).

    ``cfg`` is accepted for a uniform call signature; it is not needed.
    """
    param = next(iter(decoder.parameters()), None) if hasattr(decoder, "parameters") else None
    device = code.device if param is None else param.device
    z = code.detach().reshape(-1).to(device)
    z = z / z.norm()
    dim = z.shape[0]

    # 1. surface points
    x = torch.rand(points, 3, generator=generator, device=generator.device if generator is not None else "cpu")
    x = (x * 2 - 1).to(device=device, dtype=z.dtype)
    with torch.no_grad():
        mod = decoder.prepare(z[None])
    f, g = _field_and_grad(decoder, mod, x)
    xp = x - f[:, None] * g
    with torch.no_grad():
        fp_all = decoder.query(mod, xp[None])[0]
    fp_abs = fp_all.abs()
    finite = torch.isfinite(fp_abs) & (xp.abs().amax(dim=1) <= 1.0)
    e = float(fp_abs[finite].median()) if bool(finite.any()) else float("nan")
    threshold = max(e, min_motion) if math.isfinite(e) else float("inf")
    if not math.isfinite(e):
        return _empty_result(dim, e, threshold, 0)
    keep = finite & (fp_abs <= max(surface_factor * e, surface_floor))
    p = xp[keep]
    _, gp = _field_and_grad(decoder, mod, p)
    gnorm = gp.norm(dim=1)
    ok = gnorm > 1e-6
    p, gp, gnorm = p[ok], gp[ok], gnorm[ok]
    count = int(p.shape[0])
    if count < 8:
        return _empty_result(dim, e, threshold, count)
    normal = -gp / gnorm[:, None]

    with torch.no_grad():
        # 2. displacement per unit chord along each tangent direction
        basis = _tangent_basis(z).to(device=device, dtype=z.dtype)  # (D-1)×D
        plus = z[None] + step * basis
        minus = z[None] - step * basis
        stepped = torch.cat([plus, minus], dim=0)
        stepped = stepped / stepped.norm(dim=1, keepdim=True)
        values = []
        for start in range(0, stepped.shape[0], code_chunk):
            block = stepped[start : start + code_chunk]
            values.append(decoder.query(decoder.prepare(block), p[None].expand(block.shape[0], -1, -1)))
        values = torch.cat(values, dim=0).double()  # 2(D-1) × S
        df = (values[: dim - 1] - values[dim - 1 :]) / (2.0 * step)
        disp = (df / gnorm[None].double()).T  # S × (D-1)

        # 3. remove translation, rotation and uniform scale
        pd, nd = p.double(), normal.double()
        pose = torch.cat([nd, torch.linalg.cross(pd, nd, dim=1), (nd * pd).sum(dim=1, keepdim=True)], dim=1)
        # n·(e_i × p) = e_i·(p × n). Columns that vanish (rotations of a sphere)
        # are dropped by the rank cut-off.
        u, s_pose, _ = torch.linalg.svd(pose, full_matrices=False)
        u = u[:, s_pose > 1e-6 * s_pose.max()]
        total = float((disp**2).sum())
        residual = disp - u @ (u.T @ disp)
        after_pose = float((residual**2).sum())
        tiny = 1e-30
        pose_share = 1.0 - after_pose / total if total > tiny else 0.0

        # 4. smooth over the surface at the planner's resolution
        smooth = _neighbour_mean(pd, residual, planner_resolution)
        after_smooth = float((smooth**2).sum())
        s_raw = torch.linalg.svdvals(residual / math.sqrt(count))
        # When nothing is left after pose removal but finite-difference noise
        # (strongest direction moves less than min_motion over a cluster), the
        # ratio below would be noise over noise; report 0.
        measurable = after_pose > tiny and float(s_raw[0]) * cluster_radius > min_motion
        wobble_share = 1.0 - after_smooth / after_pose if measurable else 0.0

        # 5. singular values = RMS displacement per unit chord
        _, s, vh = torch.linalg.svd(smooth / math.sqrt(count), full_matrices=False)
        directions = vh @ basis.double()  # (D-1)×D, unit tangent vectors

    return {
        "singular_values": s.float().cpu(),
        "singular_values_unsmoothed": s_raw.float().cpu(),
        "directions": directions.float().cpu(),
        "live": int((s * cluster_radius > threshold).sum()),
        "pose_share": float(pose_share),
        "wobble_share": float(wobble_share),
        "consistency_error": e,
        "threshold": float(threshold),
        "rms_motion": math.sqrt(total / (count * (dim - 1))),
        "surface_points": count,
    }


def subspace_stability(result_a: Mapping, result_b: Mapping, k: Optional[int] = None) -> float:
    """Overlap of the live subspaces of two nearby codes, in [0, 1].

    Mean squared cosine between the top-``k`` directions of each result,
    ``||V_a^T V_b||_F^2 / k``. ``k`` defaults to the smaller of the two live
    counts. 1 means the same subspace, about ``k / (D-1)`` is chance. Returns
    NaN when ``k`` is 0 (nothing to compare).
    """
    if k is None:
        k = min(int(result_a["live"]), int(result_b["live"]))
    k = min(int(k), result_a["directions"].shape[0], result_b["directions"].shape[0])
    if k <= 0:
        return float("nan")
    va = result_a["directions"][:k].double()
    vb = result_b["directions"][:k].double()
    return float(((va @ vb.T) ** 2).sum()) / k


def summarise(results: Sequence[Mapping]) -> dict:
    """Aggregate ``dead_directions`` results over probe codes into log scalars."""
    if len(results) == 0:
        return {"probes": 0}
    live = [float(r["live"]) for r in results]
    top = [float(r["singular_values"][0]) if len(r["singular_values"]) else 0.0 for r in results]

    def mean(key: str) -> float:
        return float(np.mean([float(r[key]) for r in results]))

    # Effective number of directions: how many latent directions the surface motion is really
    # spread over (exp of the entropy of the variance shares; 1 = a single dial). Also the
    # share of the variance in the strongest direction, and the 5th strongest's motion.
    effective, top_share, fifth = [], [], []
    for r in results:
        s = np.asarray([float(v) for v in r["singular_values"]], dtype=np.float64)
        var = s ** 2
        if var.size == 0 or var.sum() <= 0:
            effective.append(0.0); top_share.append(0.0); fifth.append(0.0)
            continue
        p = var / var.sum()
        effective.append(float(np.exp(-(p * np.log(p + 1e-30)).sum())))
        top_share.append(float(p.max()))
        fifth.append(float(np.sort(s)[::-1][4]) if s.size >= 5 else 0.0)

    return {
        "probes": len(results),
        "effective_directions_mean": float(np.mean(effective)),
        "top_direction_share_mean": float(np.mean(top_share)),
        "fifth_direction_motion_mean": float(np.mean(fifth)),
        "live_mean": float(np.mean(live)),
        "live_min": float(np.min(live)),
        "live_max": float(np.max(live)),
        "wobble_share_mean": mean("wobble_share"),
        "pose_share_mean": mean("pose_share"),
        "consistency_error_mean": mean("consistency_error"),
        "top_singular_value_mean": float(np.mean(top)),
        "surface_points_mean": mean("surface_points"),
    }
