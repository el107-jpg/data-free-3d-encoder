"""Geometric descriptors of a solid given as a field on a cubic grid over [-1, 1]^3 (positive inside).

They describe the KIND of geometry, not where the volume sits:

* ``linear``, ``planar``, ``round`` (sum to 1): how rod-like, sheet-like or blob-like the solid is locally,
  averaged over its volume. Around each sampled inside point the inside points within three local
  thicknesses (six times the local depth of the skeleton) are collected and the spreads s1 >= s2 >= s3 along their principal axes give
  (s1 - s2) / s1, (s2 - s3) / s1 and s3 / s1.
* ``thickness``: twice the median depth on the skeleton (the ridge of the field).
* ``ray_crossings`` and ``star_fraction``: rays leave the centroid in all directions; the mean number of
  times a ray crosses the surface, and the share of rays that cross exactly once. A star-shaped solid
  has 1 and 1. A solid that folds back around its centre has more crossings and a smaller share.
* ``fold``: longest shortest-path inside the solid divided by the longest straight-line distance between
  two of its points. 1 for convex solids (the voxel paths add up to about 8%), larger for bent ones.
* ``solidity``: volume divided by the volume of the convex hull.
* ``branches``: the solid is cut into bands of equal inside-distance from one end; the largest number of
  separate pieces in a band (1 for a rod or a plate, 2 for a Y or a T seen from one arm, and so on).
* ``gap``: width of the narrowest gap that closes into a handle or a cavity when the solid is grown
  (0 when growing it by up to ``gap_limit`` changes nothing). A C is "nearly a ring" in this sense.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import scipy.ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.distance import pdist

KEYS = ("linear", "planar", "round", "thickness", "ray_crossings", "star_fraction", "fold", "solidity",
        "branches", "gap")


def _simple(mask: np.ndarray) -> bool:
    from arm_planner.data_free.diagnostics import _topology_of_mask
    topo = _topology_of_mask(mask)
    return topo["pieces"] == 1 and topo["cavities"] == 0 and topo["handles"] == 0


def fibonacci_sphere(count: int) -> np.ndarray:
    k = np.arange(count) + 0.5
    z = 1.0 - 2.0 * k / count
    phi = np.pi * (1.0 + 5.0 ** 0.5) * k
    r = np.sqrt(1.0 - z * z)
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], 1)


def _voxel_graph(solid: np.ndarray, h: float):
    index = -np.ones(solid.shape, dtype=np.int64)
    count = int(solid.sum())
    index[solid] = np.arange(count)
    rows, cols, weights = [], [], []
    n = solid.shape[0]
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if (dx, dy, dz) <= (0, 0, 0):
                    continue                                          # each pair once
                src = tuple(slice(max(0, -d), n - max(0, d)) for d in (dx, dy, dz))
                dst = tuple(slice(max(0, d), n - max(0, -d)) for d in (dx, dy, dz))
                a, b = index[src], index[dst]
                ok = (a >= 0) & (b >= 0)
                rows.append(a[ok]); cols.append(b[ok])
                weights.append(np.full(int(ok.sum()), h * (dx * dx + dy * dy + dz * dz) ** 0.5))
    rows, cols, weights = np.concatenate(rows), np.concatenate(cols), np.concatenate(weights)
    return coo_matrix((weights, (rows, cols)), shape=(count, count)).tocsr()


def describe(grid: np.ndarray, soft_tau: float = 0.02, seed: int = 0, samples: int = 300,
             rays: int = 1500, gap_limit: float = 0.24, gap_step: float = 0.02) -> Optional[Dict[str, float]]:
    """Descriptors of the solid ``grid > 0``; None when the solid is empty."""
    grid = np.asarray(grid, dtype=np.float32)
    res = grid.shape[0]
    h = 2.0 / (res - 1)
    axis = np.linspace(-1.0, 1.0, res)
    solid = grid > 0
    if solid.sum() < 30:
        return None
    rng = np.random.default_rng(seed)
    ii, jj, kk = np.nonzero(solid)
    inside = np.stack([axis[ii], axis[jj], axis[kk]], 1)
    depth = grid[solid]

    # skeleton: where the field stops rising
    gx, gy, gz = np.gradient(grid, h)
    ridge = (np.sqrt(gx * gx + gy * gy + gz * gz) < 0.6)[solid]
    if ridge.sum() < 1:
        ridge = depth >= depth.max() - 1e-6
    ridge_points, ridge_depth = inside[ridge], depth[ridge]
    thickness = 2.0 * float(np.median(ridge_depth))

    # local rod / sheet / blob character, averaged over the volume
    thin = inside if len(inside) <= 8000 else inside[rng.choice(len(inside), 8000, replace=False)]
    tree, ridge_tree = cKDTree(thin), cKDTree(ridge_points)
    picks = inside[rng.choice(len(inside), min(samples, len(inside)), replace=False)]
    _, near = ridge_tree.query(picks)
    shares = []
    for x, m in zip(picks, near):
        nb = thin[tree.query_ball_point(x, max(6.0 * float(ridge_depth[m]), 4.0 * h))]
        if len(nb) < 12:
            continue
        s = np.sqrt(np.clip(np.linalg.eigvalsh(np.cov(nb.T)), 0.0, None))[::-1]
        if s[0] <= 0:
            continue
        shares.append([(s[0] - s[1]) / s[0], (s[1] - s[2]) / s[0], s[2] / s[0]])
    linear, planar, round_ = (np.mean(shares, 0) if shares else np.array([0.0, 0.0, 1.0]))

    # rays from the (soft) centroid
    weight = 1.0 / (1.0 + np.exp(-np.clip(grid / soft_tau, -60, 60)))
    total = weight.sum()
    centroid = np.array([(weight.sum((1, 2)) * axis).sum(), (weight.sum((0, 2)) * axis).sum(),
                         (weight.sum((0, 1)) * axis).sum()]) / total
    t = np.linspace(0.0, 1.6, 320)
    points = centroid + t[None, :, None] * fibonacci_sphere(rays)[:, None, :]
    coords = ((points + 1.0) / 2.0 * (res - 1)).reshape(-1, 3).T
    values = ndi.map_coordinates(grid, coords, order=1, mode="constant", cval=-1.0).reshape(rays, -1)
    side = values > 0
    crossings = (side[:, 1:] != side[:, :-1]).sum(1)

    # inside distances: two sweeps find the two ends of the solid
    graph = _voxel_graph(solid, h)
    start = int(np.argmax(((inside - centroid) ** 2).sum(1)))
    first = dijkstra(graph, directed=False, indices=start)
    first[~np.isfinite(first)] = -1.0
    end_a = int(np.argmax(first))
    second = dijkstra(graph, directed=False, indices=end_a)
    reach = np.isfinite(second)
    geodesic = float(second[reach].max())
    hull = ConvexHull(inside)
    straight = float(pdist(inside[hull.vertices]).max())
    # the hull of the voxel CUBES, not of their centres (which loses half a voxel all round a thin solid)
    corners = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]) * (0.5 * h)
    hull = ConvexHull((inside[hull.vertices][:, None, :] + corners[None]).reshape(-1, 3))
    bands = np.minimum((np.where(reach, second, 0.0) / (geodesic + 1e-9) * 10).astype(np.int64), 9)
    branches, label_grid = 1, np.zeros(solid.shape, dtype=bool)
    for band in range(10):
        member = reach & (bands == band)
        if member.sum() < 4:
            continue
        label_grid[...] = False
        label_grid[ii[member], jj[member], kk[member]] = True
        labels, pieces = ndi.label(label_grid, structure=np.ones((3, 3, 3), dtype=bool))
        sizes = np.bincount(labels.ravel())[1:]
        branches = max(branches, int((sizes >= max(4, 0.08 * member.sum())).sum()))

    # near-topology: how much growth closes a gap into a handle or a cavity
    gap = 0.0
    if _simple(solid):
        grow = gap_step
        while grow <= gap_limit + 1e-9:
            if not _simple(grid > -grow):
                gap = 2.0 * grow
                break
            grow += gap_step
    return {"linear": float(linear), "planar": float(planar), "round": float(round_), "thickness": thickness,
            "ray_crossings": float(crossings.mean()), "star_fraction": float((crossings == 1).mean()),
            "fold": geodesic / straight, "solidity": float(solid.sum() * h ** 3 / hull.volume),
            "branches": float(branches), "gap": gap}


def vector(d: Dict[str, float]) -> np.ndarray:
    """Descriptor as a vector for distances; the gap enters as its closeness (wide or no gap -> 0)."""
    v = np.array([d[k] for k in KEYS], dtype=np.float64)
    v[KEYS.index("gap")] = 0.0 if d["gap"] <= 0 else max(0.0, 1.0 - d["gap"] / 0.5)
    return v
