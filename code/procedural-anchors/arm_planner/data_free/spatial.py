"""Batched spatial primitives for the point graph: hash-grid pairs, segments, components.

Everything here works on flat tensors that hold all shapes of a batch at once.
A point's shape is an integer id; pairs are only formed inside one shape.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

# Largest number of candidate pairs expanded at once (before the distance
# filter). Larger searches are split into a few chunks over the query points.
_CANDIDATE_CHUNK = 6_000_000


def _empty_pairs(device, dtype, return_distance: bool):
    z = torch.zeros(0, dtype=torch.long, device=device)
    if return_distance:
        return z, z.clone(), torch.zeros(0, dtype=dtype, device=device)
    return z, z.clone()


def pairs_within(
    a: Tensor,
    a_shape: Tensor,
    b: Tensor,
    b_shape: Tensor,
    radius: float,
    max_pairs: Optional[int] = None,
    *,
    unordered: bool = False,
    a_radius: Optional[Tensor] = None,
    b_radius: Optional[Tensor] = None,
    return_distance: bool = False,
    cells: int = 1,
):
    """All pairs ``(ia, ib)`` with ``a_shape[ia] == b_shape[ib]`` and distance ≤ ``radius``.

    Hash grid: integer cell keys that include the shape id, one sort of ``b``,
    ``searchsorted`` over the 27 neighbouring cells of every ``a`` (done as 9
    ranges, because the three cells along z are consecutive keys), and a
    vectorised expansion of the ranges. No k-d tree and no dense distances.

    - ``unordered=True`` is for ``a is b`` (the same point set on both sides):
      each unordered pair is returned once and no point is paired with itself.
      Only half of the neighbouring cells are visited.
    - ``a_radius`` / ``b_radius`` (per point, optional): a pair is kept when its
      distance is at most ``a_radius[ia] + b_radius[ib]`` (a missing one counts
      as zero) and also at most ``radius``. ``radius`` is then the upper bound
      that sets the cell size.
    - ``max_pairs``: if more pairs are found, only the ``max_pairs`` closest
      are returned.
    - ``return_distance=True`` also returns the pair distances.
    - ``cells``: grid cells per ``radius`` (default 1: the 27-cell stencil).
      With 2 or 3 the stencil follows the search ball more closely, which
      pays off when ``a`` is small and ``b`` is dense (fewer candidates, more
      range lookups per ``a`` point). Not used with ``unordered``.

    Points must be finite. ``a`` is ``A×3``, ``b`` is ``B×3``; shape ids are
    non-negative integers.
    """
    device, dtype = a.device, a.dtype
    radius = float(radius)
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    na, nb = a.shape[0], b.shape[0]
    if na == 0 or nb == 0:
        return _empty_pairs(device, dtype, return_distance)
    a_shape = a_shape.to(torch.long)
    b_shape = b_shape.to(torch.long)

    k = 1 if unordered else max(1, int(cells))
    cell = radius / k
    lo = torch.minimum(a.amin(dim=0), b.amin(dim=0))
    hi = torch.maximum(a.amax(dim=0), b.amax(dim=0))
    # cells are offset by k so that the -k neighbour is still a valid index
    dims = torch.floor((hi - lo) / cell).to(torch.long) + 2 * k + 1
    nx, ny, nz = (int(v) for v in dims.tolist())
    n_shapes = int(torch.maximum(a_shape.max(), b_shape.max())) + 1
    if n_shapes * nx * ny * nz >= 2 ** 62:
        raise ValueError("hash grid too large: points too spread out for this radius")

    ca = torch.floor((a - lo) / cell).to(torch.long) + k
    if unordered:
        cb = ca
    else:
        cb = torch.floor((b - lo) / cell).to(torch.long) + k
    col_a = (a_shape * nx + ca[:, 0]) * ny + ca[:, 1]  # key of the (shape, x, y) column
    key_b = ((b_shape * nx + cb[:, 0]) * ny + cb[:, 1]) * nz + cb[:, 2]
    key_sorted, order = torch.sort(key_b)

    if unordered:
        # half neighbourhood: 4 full columns ahead, plus the own column from
        # the own cell upward. Every unordered pair of cells is visited once.
        dxy = torch.tensor([[1, -1], [1, 0], [1, 1], [0, 1], [0, 0]], device=device)
        z_lo = torch.tensor([-1, -1, -1, -1, 0], device=device)
    else:
        span = range(-k, k + 1)
        # columns whose nearest corner is farther than the radius are skipped
        offs = [[i, j] for i in span for j in span
                if max(abs(i) - 1, 0) ** 2 + max(abs(j) - 1, 0) ** 2 <= k * k]
        dxy = torch.tensor(offs, device=device)
        z_lo = torch.full((len(offs),), -k, device=device)
    nq = dxy.shape[0]
    col_off = dxy[:, 0] * ny + dxy[:, 1]
    k_lo = (col_a[:, None] + col_off[None, :]) * nz + ca[:, 2:3] + z_lo[None, :]
    k_hi = (col_a[:, None] + col_off[None, :]) * nz + ca[:, 2:3] + k
    start = torch.searchsorted(key_sorted, k_lo.reshape(-1))
    stop = torch.searchsorted(key_sorted, k_hi.reshape(-1), right=True)
    counts = stop - start  # A*nq
    cum = torch.cumsum(counts, dim=0)
    total = int(cum[-1])
    if total == 0:
        return _empty_pairs(device, dtype, return_distance)

    r2 = radius * radius
    key_a = col_a * nz + ca[:, 2] if unordered else None

    def expand(q0: int, q1: int):
        cnt = counts[q0:q1]
        base = cum[q0 - 1] if q0 > 0 else 0
        n = int(cum[q1 - 1] - base)
        if n == 0:
            return None
        rep = torch.repeat_interleave(torch.arange(q0, q1, device=device), cnt)
        pos = torch.arange(n, device=device) - (cum[rep] - counts[rep] - base) + start[rep]
        ia = torch.div(rep, nq, rounding_mode="floor")
        ib = order[pos]
        d2 = (a[ia] - b[ib]).square().sum(dim=-1)
        keep = d2 <= r2
        if a_radius is not None or b_radius is not None:
            thr = 0.0
            if a_radius is not None:
                thr = thr + a_radius[ia]
            if b_radius is not None:
                thr = thr + b_radius[ib]
            keep &= d2 <= thr * thr
            keep &= thr >= 0
        if unordered:
            # inside one cell both orders (and the point itself) show up
            keep &= (key_sorted[pos] != key_a[ia]) | (ib > ia)
        ia, ib = ia[keep], ib[keep]
        if return_distance or max_pairs is not None:
            return ia, ib, d2[keep].sqrt()
        return ia, ib, None

    if total <= _CANDIDATE_CHUNK:
        parts = [expand(0, na * nq)]
    else:
        # a few chunks over the queries, each with a bounded candidate count
        n_chunks = total // _CANDIDATE_CHUNK + 1
        targets = torch.arange(1, n_chunks, device=device) * (total // n_chunks)
        cuts = torch.searchsorted(cum, targets).tolist()
        bounds = [0] + [min(c + 1, na * nq) for c in cuts] + [na * nq]
        parts = [expand(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)
                 if bounds[i + 1] > bounds[i]]
    parts = [p for p in parts if p is not None]
    if not parts:
        return _empty_pairs(device, dtype, return_distance)
    if len(parts) == 1:
        ia, ib, dist = parts[0]
    else:
        ia = torch.cat([p[0] for p in parts])
        ib = torch.cat([p[1] for p in parts])
        dist = torch.cat([p[2] for p in parts]) if parts[0][2] is not None else None

    if max_pairs is not None and ia.numel() > max_pairs:
        sel = torch.topk(dist, int(max_pairs), largest=False).indices
        ia, ib, dist = ia[sel], ib[sel], dist[sel]
    if return_distance:
        return ia, ib, dist
    return ia, ib


def point_segment_distance(p: Tensor, a: Tensor, b: Tensor) -> Tuple[Tensor, Tensor]:
    """Distance from ``p`` to the segment ``a``–``b`` and the parameter ``t`` in [0, 1].

    ``t`` locates the closest point ``a + t (b − a)``. Broadcasts over leading
    dimensions; a zero-length segment gives the distance to ``a`` and ``t = 0``.
    """
    ab = b - a
    ap = p - a
    denom = ab.square().sum(dim=-1)
    t = (ap * ab).sum(dim=-1) / denom.clamp_min(1e-20)
    t = torch.where(denom > 0, t, torch.zeros_like(t)).clamp(0.0, 1.0)
    closest = a + t.unsqueeze(-1) * ab
    return (p - closest).norm(dim=-1), t


def connected_components(
    num_nodes: int, edge_a: Tensor, edge_b: Tensor, return_iterations: bool = False
):
    """Component label of every node: the smallest node index in its component.

    Scatter-min label propagation with pointer jumping (the FastSV scheme):
    each round takes, for every node, the smallest grandparent among its
    neighbours, hooks that onto the node's parent and onto the node itself,
    and shortcuts parents to grandparents. The number of rounds grows with
    the logarithm of the component diameter, not the diameter. All shapes of a
    batch are handled at once (node indices are global and there are no
    edges between shapes).
    """
    device = edge_a.device
    parent = torch.arange(num_nodes, device=device)
    if edge_a.numel() == 0 or num_nodes == 0:
        return (parent, 0) if return_iterations else parent
    src = torch.cat([edge_a, edge_b]).to(torch.long)
    dst = torch.cat([edge_b, edge_a]).to(torch.long)
    rounds = 0
    while True:
        rounds += 1
        grand = parent[parent]
        # smallest grandparent among the node itself and its neighbours
        best = grand.scatter_reduce(0, src, grand[dst], reduce="amin", include_self=True)
        # hook onto the parent (tree hooking), onto the node, and shortcut
        new = parent.scatter_reduce(0, parent, best, reduce="amin", include_self=True)
        new = torch.minimum(new, best)
        if torch.equal(new, parent):
            break
        parent = new
    return (parent, rounds) if return_iterations else parent
