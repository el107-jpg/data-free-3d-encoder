"""Connectivity, cavities and thin features from the points a training step already has.

``build_graph`` takes, per shape, the volume samples ``x`` (values ``f``,
gradients ``g``) and their one-step surface guesses ``xp`` (values ``fp``) and
decides, without a voxel grid and without any further field evaluation:

- which solid pieces exist (optimistic and pessimistic components),
- which free regions are enclosed (cavities),
- where two walls of one piece face each other across a gap narrower than the
  minimum feature size ``t`` and where a wall is thinner than ``t``.

Sign convention: positive inside. Signs are exact evidence, values are hints:
every ball radius is shrunk before use. A cut removes one edge; components
come only from label propagation. The graph is built under ``no_grad`` from
detached tensors and only selects points.

All shapes of the batch are processed together on flat tensors (point index
``shape * 2N + j``); there is no Python loop over points, edges or shapes.

Where this file is stricter than ``docs/data_free_implementation_spec.md``
(all in the "over-split rather than merge" direction) it is marked DEVIATION.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from .spatial import connected_components, pairs_within, point_segment_distance

# Points whose coordinates leave this box (or are not finite) are ignored.
_COORD_LIMIT = 2.0
# Upper edge lengths (fractions of r_s) of the cut-test stages of the solid
# graph; a last stage takes everything longer. See build_graph.
_STAGE_FRACTIONS = (0.5, 0.78)
# Grid cells per search radius in the cut tests (few query points, many edges).
_CUT_CELLS = 3
_PAIR_BLOCK = 4_000_000   # pairs handled at once in the cut tests (bounds their memory)
# Segments × points handled at once by the dense cut test, and the largest
# total for which the dense test replaces the hash grid.
_DENSE_SEGMENT_CHUNK = 2_000_000
_DENSE_SEGMENT_LIMIT = 6_000_000
# Same for the outside graph, as a fraction of r_v.
_OUT_STAGE1_FRACTION = 0.6
# Dense deep-node step: largest shapes × deep × candidates block evaluated at once.
_DENSE_BUDGET = 12_000_000


@dataclass
class ShapeGraph:
    # per point, B×2N (first N: volume samples, last N: their guesses)
    inside: Tensor
    band: Tensor
    outside: Tensor
    solid_label: Tensor       # long, optimistic component, -1 for non-nodes
    solid_label_pess: Tensor  # long, pessimistic (certified-only) component
    outside_label: Tensor     # long, optimistic component of the outside graph
    main_label: Tensor        # B, optimistic label of the main body (-1: no solid node)
    # fragments, F rows
    frag_shape: Tensor
    frag_kind: Tensor         # 0: separate in the optimistic graph, 1: only in the pessimistic
    frag_size: Tensor         # largest F among the fragment's nodes
    frag_gap: Tensor          # length of the joining segment
    frag_a: Tensor            # F×3, end on the fragment
    frag_b: Tensor            # F×3, end on the main body
    frag_mask: Tensor         # F×2N bool, member points
    # cavities, K rows
    cav_shape: Tensor
    cav_a: Tensor             # K×3, end on the cavity
    cav_b: Tensor             # K×3, end on the far-field component
    cav_mask: Tensor          # K×2N bool
    # thin sites
    gap_shape: Tensor
    gap_mid: Tensor
    gap_len: Tensor
    wall_shape: Tensor
    wall_mid: Tensor
    wall_len: Tensor
    stats: dict = field(default_factory=dict)
    # extras (not in the spec): uncapped counts per shape, and the settings used
    num_pieces: Optional[Tensor] = None     # B, optimistic pieces incl. the main body
    num_cavities: Optional[Tensor] = None   # B
    main_label_pess: Optional[Tensor] = None
    far_label: Optional[Tensor] = None      # B, outside_label of the far-field component
    delta: Optional[Tensor] = None          # B, band half-width used
    mode: str = "full"

    @property
    def num_shapes(self) -> int:
        return self.inside.shape[0]


# --------------------------------------------------------------------------- helpers


def _unpack(fb_or_tensors):
    if hasattr(fb_or_tensors, "xp") and hasattr(fb_or_tensors, "fp"):
        fb = fb_or_tensors
        return fb.x, fb.f, fb.g, fb.xp, fb.fp
    x, f, g, xp, fp = fb_or_tensors
    return x, f, g, xp, fp


def _cfg(cfg, name: str, default):
    value = getattr(cfg, name, None)
    return default if value is None else value


def _topk_per_group(group: Tensor, score: Tensor, k: int, largest: bool) -> Tensor:
    """Indices of the ``k`` best ``score`` rows of every ``group`` (grouped, best first)."""
    if group.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=group.device)
    order = torch.argsort(score, descending=largest, stable=True)
    order = order[torch.argsort(group[order], stable=True)]
    g = group[order]
    counts = torch.bincount(g)
    starts = torch.cumsum(counts, 0) - counts
    rank = torch.arange(g.numel(), device=g.device) - starts[g]
    return order[rank < k]


def _main_label(label: Tensor, num_shapes: int, per_shape: int, use_origin: bool = True) -> Tensor:
    """Label of the origin node when it is a node, else the most populous label."""
    device = label.device
    node = label >= 0
    counts = torch.bincount(label[node], minlength=num_shapes * per_shape)
    counts = counts.view(num_shapes, per_shape)
    offset = torch.arange(num_shapes, device=device) * per_shape
    biggest = counts.argmax(dim=1) + offset
    biggest = torch.where(counts.sum(dim=1) > 0, biggest, torch.full_like(biggest, -1))
    if not use_origin:
        return biggest
    origin = label[offset]
    return torch.where(origin >= 0, origin, biggest)


def _ball_cut(
    seg_a: Tensor, seg_b: Tensor, seg_len: Tensor, seg_shape: Tensor,
    pts: Tensor, pts_shape: Tensor, pts_radius: Tensor, tiers: List[float],
) -> Tensor:
    """Edges whose segment comes closer to some point than that point's radius.

    The points are split into a few radius tiers so that the hash-grid cell of
    each search fits the balls of that tier.
    """
    cut = torch.zeros(seg_len.shape[0], dtype=torch.bool, device=seg_len.device)
    if seg_len.numel() == 0 or pts.shape[0] == 0:
        return cut
    mid = 0.5 * (seg_a + seg_b)
    half = 0.5 * seg_len
    half_max = float(half.max())
    lower = 0.0
    for upper in tiers:
        sel = (pts_radius > lower) & (pts_radius <= upper)
        lower = upper
        p, s, r = pts[sel], pts_shape[sel], pts_radius[sel]
        if p.shape[0] == 0:
            continue
        ip, ie = pairs_within(p, s, mid, seg_shape, half_max + upper, a_radius=r, b_radius=half,
                              cells=_CUT_CELLS)
        if ip.numel() == 0:
            continue
        for start in range(0, ip.numel(), _PAIR_BLOCK):
            jp, je = ip[start:start + _PAIR_BLOCK], ie[start:start + _PAIR_BLOCK]
            dist, _ = point_segment_distance(p[jp], seg_a[je], seg_b[je])
            cut[je[dist < r[jp]]] = True
    return cut


def _surfel_cut(
    seg_a: Tensor, seg_b: Tensor, seg_len: Tensor, seg_shape: Tensor,
    s_pts: Tensor, s_shape: Tensor, s_normal: Tensor, tol: float, cos_cut: float,
) -> Tensor:
    """C2b: a surfel within ``tol`` of the segment, ``tol`` away from both ends
    along it, whose normal is aligned with the edge."""
    cut = torch.zeros(seg_len.shape[0], dtype=torch.bool, device=seg_len.device)
    if seg_len.numel() == 0 or s_pts.shape[0] == 0:
        return cut
    sel = (seg_len >= 2.0 * tol).nonzero().squeeze(1)
    if sel.numel() == 0:
        return cut
    a, b, length, shape = seg_a[sel], seg_b[sel], seg_len[sel], seg_shape[sel]
    reach = ((0.5 * length - tol).square() + tol * tol).sqrt()
    i_s, i_e = pairs_within(
        s_pts, s_shape, 0.5 * (a + b), shape, float(reach.max()) + 1e-6, b_radius=reach,
        cells=_CUT_CELLS,
    )
    if i_s.numel() == 0:
        return cut
    # In blocks: a burst of thin features can produce tens of millions of (surfel, edge) pairs,
    # and the per-pair intermediates then exhaust the GPU (this crashed a run).
    for start in range(0, i_s.numel(), _PAIR_BLOCK):
        j_s, j_e = i_s[start:start + _PAIR_BLOCK], i_e[start:start + _PAIR_BLOCK]
        dist, t = point_segment_distance(s_pts[j_s], a[j_e], b[j_e])
        along = t * length[j_e]
        u = (b[j_e] - a[j_e]) / length[j_e].unsqueeze(-1)
        aligned = (s_normal[j_s] * u).sum(dim=-1).abs() >= cos_cut
        hit = (dist < tol) & (along >= tol) & (along <= length[j_e] - tol) & aligned
        cut[sel[j_e[hit]]] = True
    return cut


def _node_ball_relation(
    node_ids: Tensor, pts: Tensor, shape: Tensor, total: int,
    b_pts: Tensor, b_shape: Tensor, b_radius: Tensor, tiers: List[float], half_len: float,
):
    """Which nodes lie inside a ball, and which (ball, node) pairs nearly do.

    A segment of half-length at most ``half_len`` can enter a ball only if one
    of its ends is inside the ball or within ``sqrt(r² + half_len²)`` of its
    centre ("grazing"). For short edges the grazing margin is tiny, so this
    node search returns far fewer pairs than a search over edge midpoints.
    """
    device = pts.device
    in_ball = torch.zeros(total, dtype=torch.bool, device=device)
    graze_ball, graze_node = [], []
    n_pts, n_shape = pts[node_ids], shape[node_ids]
    lower = 0.0
    for upper in tiers:
        sel = ((b_radius > lower) & (b_radius <= upper)).nonzero().squeeze(1)
        lower = upper
        if sel.numel() == 0 or node_ids.numel() == 0:
            continue
        r = b_radius[sel]
        reach = (r.square() + half_len * half_len).sqrt()
        ip, jn, dist = pairs_within(
            b_pts[sel], b_shape[sel], n_pts, n_shape, (upper * upper + half_len * half_len) ** 0.5,
            a_radius=reach, return_distance=True, cells=_CUT_CELLS,
        )
        inside = dist < r[ip]
        in_ball[node_ids[jn[inside]]] = True
        graze_ball.append(sel[ip[~inside]])
        graze_node.append(node_ids[jn[~inside]])
    empty = torch.zeros(0, dtype=torch.long, device=device)
    return (in_ball, torch.cat(graze_ball) if graze_ball else empty,
            torch.cat(graze_node) if graze_node else empty)


def _ball_cut_by_nodes(relation, ea: Tensor, eb: Tensor, pts: Tensor,
                       b_pts: Tensor, b_radius: Tensor) -> Tensor:
    """C1 for short edges from a node–ball relation (see ``_node_ball_relation``)."""
    in_ball, graze_ball, graze_node = relation
    cut = in_ball[ea] | in_ball[eb]
    n_edges = ea.numel()
    if graze_ball.numel() == 0 or n_edges == 0:
        return cut
    # edges incident to each grazing node, by searching the sorted end list
    ends = torch.cat([ea, eb])
    ends_sorted, order = torch.sort(ends)
    start = torch.searchsorted(ends_sorted, graze_node)
    stop = torch.searchsorted(ends_sorted, graze_node, right=True)
    counts = stop - start
    cum = torch.cumsum(counts, 0)
    n = int(cum[-1])
    if n == 0:
        return cut
    pair = torch.repeat_interleave(torch.arange(counts.numel(), device=ea.device), counts)
    pos = torch.arange(n, device=ea.device) - (cum - counts)[pair] + start[pair]
    edge = order[pos] % n_edges
    ball = graze_ball[pair]
    dist, _ = point_segment_distance(b_pts[ball], pts[ea[edge]], pts[eb[edge]])
    cut[edge[dist < b_radius[ball]]] = True
    return cut


def _dense_cuts(
    seg_a: Tensor, seg_b: Tensor, seg_len: Tensor, seg_shape: Tensor,
    pts2: Tensor, ball_radius2: Tensor, surfel2: Tensor, normal2: Tensor,
    tol: float, cos_cut: float,
) -> Tuple[Tensor, Tensor]:
    """C1 and C2b of a few segments against every point of their shape.

    ``pts2`` is ``B×M×3``; ``ball_radius2`` (``B×M``) is zero where a point
    has no ball. Used when there are too few segments to pay for a hash grid.
    """
    n = seg_len.shape[0]
    cut_ball = torch.zeros(n, dtype=torch.bool, device=seg_len.device)
    cut_surf = torch.zeros_like(cut_ball)
    step = max(1, _DENSE_SEGMENT_CHUNK // pts2.shape[1])
    for i in range(0, n, step):  # a few chunks at most
        a, b, length, sh = seg_a[i:i + step], seg_b[i:i + step], seg_len[i:i + step], seg_shape[i:i + step]
        dist, t = point_segment_distance(pts2[sh], a[:, None, :], b[:, None, :])
        cut_ball[i:i + step] = (dist < ball_radius2[sh]).any(dim=1)
        along = t * length[:, None]
        u = (b - a) / length.clamp_min(1e-12).unsqueeze(-1)
        hit = surfel2[sh] & (dist < tol) & (along >= tol) & (along <= length[:, None] - tol)
        hit &= (normal2[sh] * u[:, None, :]).sum(dim=-1).abs() >= cos_cut
        cut_surf[i:i + step] = hit.any(dim=1)
    return cut_ball, cut_surf


def _deep_edges(
    points: Tensor, shape: Tensor, rho: Tensor, deep: Tensor, has_ball: Tensor,
    num_shapes: int, r_v: float, gap_res: Tensor,
):
    """Dense step over the few deep nodes of every shape.

    Returns certified edges (ball overlap between a deep node and any inside
    node with a ball: family 2 certified and family 4 containment) and the
    uncertified deep–deep pairs within ``r_v`` whose hidden gap is small
    enough to keep as candidates.
    """
    device = points.device
    empty = torch.zeros(0, dtype=torch.long, device=device)
    out = dict(cert_a=empty, cert_b=empty, cert_len=points.new_zeros(0), cand_a=empty,
               cand_b=empty, cand_len=points.new_zeros(0), n_vol_cert=0, n_vol_dropped=0,
               n_deep_max=0)
    d_idx = deep.nonzero().squeeze(1)
    k_idx = has_ball.nonzero().squeeze(1)
    if d_idx.numel() == 0 or k_idx.numel() == 0:
        return out

    def pad(idx):
        sh = shape[idx]
        counts = torch.bincount(sh, minlength=num_shapes)
        width = int(counts.max())
        rank = torch.arange(idx.numel(), device=device) - (torch.cumsum(counts, 0) - counts)[sh]
        table = torch.full((num_shapes, width), -1, dtype=torch.long, device=device)
        table[sh, rank] = idx
        return table, width

    d_pad, d_w = pad(d_idx)
    k_pad, k_w = pad(k_idx)
    out["n_deep_max"] = d_w
    chunk = max(1, _DENSE_BUDGET // max(1, d_w * k_w))
    cert_a, cert_b, cert_len, cand_a, cand_b, cand_len = [], [], [], [], [], []
    n_vol_cert = torch.zeros((), dtype=torch.long, device=device)
    n_vol_drop = torch.zeros((), dtype=torch.long, device=device)
    for s0 in range(0, num_shapes, chunk):  # a few chunks of shapes, usually one
        dp, kp = d_pad[s0:s0 + chunk], k_pad[s0:s0 + chunk]
        dv, kv = dp >= 0, kp >= 0
        dpc, kpc = dp.clamp_min(0), kp.clamp_min(0)
        # (matrix-product distances: error ~1e-6 here, far below any threshold)
        dist = torch.cdist(points[dpc], points[kpc])
        gap = dist - rho[dpc][:, :, None] - rho[kpc][:, None, :]
        k_deep = deep[kpc] & kv
        id_d, id_k = dp[:, :, None], kp[:, None, :]
        # deep–deep pairs appear twice in the table: keep one orientation
        keep = dv[:, :, None] & kv[:, None, :] & (id_d != id_k)
        keep &= (id_d < id_k) | ~k_deep[:, None, :]
        vol = k_deep[:, None, :] & (dist <= r_v)
        # everything of interest in one pass, classified on the sparse result
        hit = (keep & ((gap <= 0) | vol)).nonzero(as_tuple=True)
        h_gap, h_vol = gap[hit], vol[hit]
        h_a, h_b = dp[hit[0], hit[1]], kp[hit[0], hit[2]]
        is_cert = h_gap <= 0
        is_cand = ~is_cert & (h_gap <= gap_res[s0:s0 + chunk][hit[0]])
        n_vol_cert += (is_cert & h_vol).sum()
        n_vol_drop += (~is_cert & ~is_cand).sum()
        cert_a.append(h_a[is_cert])
        cert_b.append(h_b[is_cert])
        cert_len.append(dist[hit][is_cert])
        cand_a.append(h_a[is_cand])
        cand_b.append(h_b[is_cand])
        cand_len.append(dist[hit][is_cand])
    out.update(cert_a=torch.cat(cert_a), cert_b=torch.cat(cert_b), cert_len=torch.cat(cert_len),
               cand_a=torch.cat(cand_a), cand_b=torch.cat(cand_b),
               cand_len=torch.cat(cand_len), n_vol_cert=n_vol_cert, n_vol_dropped=n_vol_drop)
    return out


def _closest_pairs(points: Tensor, row_shape: Tensor, own: Tensor, other: Tensor):
    """Closest pair between each row's member points and its counterpart set.

    ``points`` is ``B×M×3``; ``own`` and ``other`` are ``R×M`` masks. Loops
    over the (capped, usually empty) list of fragments or cavities.
    """
    n = row_shape.shape[0]
    a = points.new_full((n, 3), float("nan"))
    b = points.new_full((n, 3), float("nan"))
    gap = points.new_full((n,), float("inf"))
    node_a = torch.full((n,), -1, dtype=torch.long, device=points.device)
    node_b = node_a.clone()
    per = points.shape[1]
    for i, s in enumerate(row_shape.tolist()):
        i_own = own[i].nonzero().squeeze(1)
        i_other = other[i].nonzero().squeeze(1)
        if i_own.numel() == 0 or i_other.numel() == 0:
            continue
        p_own, p_other = points[s][i_own], points[s][i_other]
        dist = torch.cdist(p_own, p_other)
        flat = torch.argmin(dist)
        ia = torch.div(flat, dist.shape[1], rounding_mode="floor")
        ib = flat - ia * dist.shape[1]
        a[i], b[i], gap[i] = p_own[ia], p_other[ib], (p_own[ia] - p_other[ib]).norm()
        node_a[i], node_b[i] = s * per + i_own[ia], s * per + i_other[ib]
    return a, b, gap, node_a, node_b


# --------------------------------------------------------------------------- main entry


@torch.no_grad()
def build_graph(fb_or_tensors, e, cfg, mode: str = "full", profile: bool = False) -> ShapeGraph:
    """Build the point graph of every shape in the batch and report its findings.

    ``fb_or_tensors``: a ``FieldBatch`` or the tuple ``(x, f, g, xp, fp)`` with
    shapes ``B×N×3, B×N, B×N×3, B×N×3, B×N``; ``x[:, 0]`` is the origin.
    ``e``: consistency error (float, 0-dim tensor, or one value per shape).
    ``mode``: ``'full'`` (balls trusted after shrinking) or ``'sign'`` (no
    ball certifies anything; cuts use a fixed small radius).
    ``profile=True`` synchronises CUDA between stages and stores per-stage
    times in ``stats`` (slower; off in training).
    """
    if mode not in ("full", "sign"):
        raise ValueError("mode must be 'full' or 'sign'")
    full = mode == "full"
    x, f, g, xp, fp = (v.detach() for v in _unpack(fb_or_tensors))
    device = x.device
    num_shapes, n = f.shape
    per = 2 * n
    total = num_shapes * per
    dtype = torch.float32

    timings: Dict[str, float] = {}
    counts: Dict[str, object] = {}
    clock = [time.perf_counter()]

    def tick(name: str) -> None:
        if not profile:
            return
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        now = time.perf_counter()
        timings[name] = timings.get(name, 0.0) + (now - clock[0]) * 1e3
        clock[0] = now

    t_start = time.perf_counter()

    # ---- settings
    t_feat = float(_cfg(cfg, "min_feature", 0.08))
    r_s = float(cfg.r_s())
    r_v = float(cfg.r_v())
    tol_c = r_s / 2.5
    cos_cut = float(_cfg(cfg, "cos_cut", 0.7))
    facing_cos = float(_cfg(cfg, "facing_cos", -0.5))
    bound_radius = float(_cfg(cfg, "bound_radius", 0.9))
    max_sites = int(_cfg(cfg, "max_sites", 32))
    max_fragments = int(_cfg(cfg, "max_fragments", 8))
    max_cavities = int(_cfg(cfg, "max_cavities", 8))
    # DEVIATION knobs (read from the config when present)
    cos_one = float(_cfg(cfg, "graph_one_sided_cos", 0.3))
    one_sided = bool(_cfg(cfg, "graph_one_sided_cut", True))
    hidden_gap = bool(_cfg(cfg, "graph_hidden_gap_drop", True))
    kind1_core = float(_cfg(cfg, "graph_kind1_core", r_s))
    rescue_degree = int(_cfg(cfg, "graph_rescue_degree", 4))
    rescue_cos = float(_cfg(cfg, "graph_rescue_cos", 0.05))
    debug = bool(_cfg(cfg, "graph_debug", False))

    e_b = torch.as_tensor(e, dtype=dtype, device=device).detach().reshape(-1)
    e_b = e_b.expand(num_shapes) if e_b.numel() == 1 else e_b
    delta_b = torch.clamp_min(
        float(_cfg(cfg, "delta_scale", 2.0)) * e_b, float(_cfg(cfg, "delta_min", 0.01))
    )
    # Smallest free gap that is resolved as a gap. Narrower ones count as contact.
    gap_res_b = torch.clamp_min(
        2.0 * delta_b, float(_cfg(cfg, "graph_gap_resolution", 0.5 * t_feat))
    )

    # ---- points and node classes
    pts = torch.cat([x, xp], dim=1).reshape(total, 3).to(dtype)
    val = torch.cat([f, fp], dim=1).reshape(total).to(dtype)
    grad = torch.cat([g, g], dim=1).reshape(total, 3).to(dtype)
    valid = (
        torch.isfinite(pts).all(dim=-1) & torch.isfinite(val) & torch.isfinite(grad).all(dim=-1)
    )
    valid &= pts.abs().amax(dim=-1) <= _COORD_LIMIT
    pts = torch.where(valid[:, None], pts, torch.zeros_like(pts))
    val = torch.where(valid, val, torch.zeros_like(val))
    grad = torch.where(valid[:, None], grad, torch.zeros_like(grad))
    normal = grad / grad.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    index = torch.arange(total, device=device)
    shape = torch.div(index, per, rounding_mode="floor")
    is_guess = (index - shape * per) >= n
    e_p, delta_p = e_b[shape], delta_b[shape]

    inside = valid & (val > 0)
    band = valid & (val.abs() <= delta_p)
    outside = valid & (val < -delta_p)
    solid = inside | band
    surfel = band & is_guess
    if full:
        rho_in = torch.where(inside, (val - e_p).clamp_min(0), torch.zeros_like(val))
        rho_free = torch.where(outside, (-val - e_p).clamp_min(0), torch.zeros_like(val))
        # DEVIATION: the free ball used to cut solid edges is shrunk by δ, not
        # e. Band nodes reach down to F = −δ, so a ball of radius |F| − e would
        # swallow slightly-negative band nodes of an exact field and cut all
        # their edges.
        cut_free = torch.where(outside, (-val - delta_p).clamp(0, r_v), torch.zeros_like(val))
        cut_in = rho_in.clamp_max(r_v)
        deep = inside & (rho_in >= 0.5 * r_s)
        tiers = [r_v / 8, r_v / 4, r_v / 2, r_v]
        rho_hint = rho_in
    else:
        # Nothing is certified: every ball radius is zero. The values are
        # still used as a hint for *which* deep pairs to try (see below).
        rho_in = torch.zeros_like(val)
        rho_free = torch.zeros_like(val)
        cut_free = torch.where(outside, torch.full_like(val, tol_c), torch.zeros_like(val))
        cut_in = torch.where(inside, torch.full_like(val, tol_c), torch.zeros_like(val))
        rho_hint = torch.where(inside, (val - e_p).clamp_min(0), torch.zeros_like(val))
        deep = inside & (rho_hint >= 0.5 * r_s)
        tiers = [tol_c]
    tick("node_classing")

    # ================================================================ solid graph
    # ---- family 1: surface edges among solid nodes
    s_idx = solid.nonzero().squeeze(1)
    ia, ib, s_len = pairs_within(
        pts[s_idx], shape[s_idx], pts[s_idx], shape[s_idx], r_s,
        unordered=True, return_distance=True,
    )
    ea, eb = s_idx[ia], s_idx[ib]
    tick("surface_edges")

    gap_res_used = gap_res_b if hidden_gap else torch.full_like(gap_res_b, float("inf"))

    def classify(ea: Tensor, eb: Tensor, length: Tensor, rescue: bool = False):
        """Certified / locally cut / candidate state of surface edges, and site flags."""
        pa, pb = pts[ea], pts[eb]
        u = (pb - pa) / length.clamp_min(1e-12).unsqueeze(-1)
        dot_a = (u * normal[ea]).sum(dim=-1)   # > 0: the edge enters the solid at a
        dot_b = (u * normal[eb]).sum(dim=-1)   # > 0: the edge arrives at b from free space
        band_a, band_b = band[ea], band[eb]
        hidden = length - rho_in[ea] - rho_in[eb]
        if full:
            cert = inside[ea] & inside[eb] & (hidden <= 0)
        else:
            cert = torch.zeros_like(band_a)
        both_band = band_a & band_b
        # C2a: leaves the solid at a and re-enters at b
        c2a = both_band & (dot_a < -cos_cut) & (dot_b > cos_cut)
        res = gap_res_b[shape[ea]]
        cut = c2a.clone()
        if one_sided:
            # DEVIATION (stricter): one end is enough. At a band node the
            # tangent plane is known; if the other end lies on the free side
            # of it by more than the gap resolution, the edge crosses free space.
            cut |= band_a & (dot_a < -cos_one) & (-length * dot_a > res)
            cut |= band_b & (dot_b > cos_one) & (length * dot_b > res)
        if hidden_gap:
            # DEVIATION (stricter): between two nodes that are not on the
            # surface, an uncovered length above the gap resolution may hide a
            # real gap and is no evidence of connection.
            cut |= ~band_a & ~band_b & (hidden > res)
        if rescue:
            # A long rescue edge may not leave the solid at a band end at all
            # (beyond a small angular tolerance): it has to head into the
            # solid half-space of every surface point it touches.
            cut |= (band_a & (dot_a < -rescue_cos)) | (band_b & (dot_b > rescue_cos))
        cut &= ~cert
        facing = both_band & ((normal[ea] * normal[eb]).sum(dim=-1) < facing_cos)
        facing &= length < t_feat
        gap_site = facing & c2a
        wall_site = facing & (dot_a > cos_cut) & (dot_b < -cos_cut)
        return cert, cut, c2a & ~cert, gap_site, wall_site

    s_cert, local_cut, c2a, gap_site, wall_site = classify(ea, eb, s_len)
    n_primary = ea.numel()
    if rescue_degree > 0:
        # DEVIATION (extension of family 1): r_s is sized for the surface
        # density of the initial sphere. Where the surface is sampled more
        # sparsely a node can have no neighbour within r_s at all and would
        # show up as a fragment. Nodes with fewer than `rescue_degree` usable
        # edges also get edges out to 2 r_s; these go through every cut test.
        usable = ~local_cut
        degree = torch.bincount(torch.cat([ea[usable], eb[usable]]), minlength=total)
        low = solid & (degree < rescue_degree)
        l_idx = low.nonzero().squeeze(1)
        ka, kb, k_len = pairs_within(
            pts[l_idx], shape[l_idx], pts[s_idx], shape[s_idx], 2.0 * r_s, return_distance=True
        )
        xa, xb = l_idx[ka], s_idx[kb]
        new = (k_len > r_s) & (~low[xb] | (xa < xb))
        xa, xb, k_len = xa[new], xb[new], k_len[new]
        x_cert, x_cut, x_c2a, _, _ = classify(xa, xb, k_len, rescue=True)
        ea, eb, s_len = torch.cat([ea, xa]), torch.cat([eb, xb]), torch.cat([s_len, k_len])
        s_cert, local_cut = torch.cat([s_cert, x_cert]), torch.cat([local_cut, x_cut])
        c2a = torch.cat([c2a, x_c2a])
        pad = torch.zeros_like(x_cert)
        gap_site, wall_site = torch.cat([gap_site, pad]), torch.cat([wall_site, pad])
    e_shape = shape[ea]
    s_cand = ~s_cert & ~local_cut
    tick("edge_local_rules")

    # ---- families 2 and 4: deep nodes (dense, few per shape)
    deep_out = _deep_edges(pts, shape, rho_hint, deep, inside & (rho_hint > 0),
                           num_shapes, r_v, gap_res_used)
    if not full:
        # INTERPRETATION of mode='sign': the pairs that the balls would
        # certify are kept, but only as candidates that must pass the cut
        # tests. Without them an interior sample is tied to the rest of the
        # solid by its parent edge alone (the origin of the initial sphere has
        # a zero gradient, so its guess is itself and it would be isolated).
        empty = torch.zeros(0, dtype=torch.long, device=device)
        deep_out.update(
            cand_a=torch.cat([deep_out["cand_a"], deep_out["cert_a"]]),
            cand_b=torch.cat([deep_out["cand_b"], deep_out["cert_b"]]),
            cand_len=torch.cat([deep_out["cand_len"], deep_out["cert_len"]]),
            cert_a=empty, cert_b=empty, n_vol_cert=0,
        )
    tick("deep_edges")

    # ---- family 3: parent edges
    samp = (inside & ~is_guess).nonzero().squeeze(1)
    samp = samp[solid[samp + n]]
    par_a, par_b = samp, samp + n
    par_cert = band[par_b] if full else torch.zeros_like(samp, dtype=torch.bool)

    # ---- pessimistic components: certified edges only
    cert_a = torch.cat([ea[s_cert], deep_out["cert_a"], par_a[par_cert]])
    cert_b = torch.cat([eb[s_cert], deep_out["cert_b"], par_b[par_cert]])
    label_p, it_p = connected_components(total, cert_a, cert_b, return_iterations=True)
    tick("label_propagation")

    # ---- cut tests on the candidates, in stages of growing edge length.
    # A candidate whose two ends are already in one component cannot change
    # the components, so it is never tested. Stage 1 tests the shortest edges
    # (enough to hold a normally sampled surface together); each later stage
    # tests only the candidates that would still join two components. The
    # result is exactly the components of "certified + every uncut candidate".
    c_idx = s_cand.nonzero().squeeze(1)
    cand_a = torch.cat([ea[c_idx], deep_out["cand_a"]])
    cand_b = torch.cat([eb[c_idx], deep_out["cand_b"]])
    cand_len = torch.cat([s_len[c_idx], deep_out["cand_len"]])
    n_surface_cand = c_idx.numel()
    o_idx = (cut_free > 0).nonzero().squeeze(1)
    o_pts, o_shape, o_rad = pts[o_idx], shape[o_idx], cut_free[o_idx]
    f_idx = surfel.nonzero().squeeze(1)
    f_pts, f_shape, f_normal = pts[f_idx], shape[f_idx], normal[f_idx]

    pts2 = pts.view(num_shapes, per, 3)
    normal2 = normal.view(num_shapes, per, 3)
    surfel2 = surfel.view(num_shapes, per)
    cut_free2 = cut_free.view(num_shapes, per)

    def cut_tests(ka, kb, kl, b_pts, b_shape, b_rad, b_rad2):
        """C1 against the given balls and C2b, dense when there are few edges."""
        ksh = shape[ka]
        if ka.numel() * per <= _DENSE_SEGMENT_LIMIT:
            return _dense_cuts(pts[ka], pts[kb], kl, ksh, pts2, b_rad2, surfel2, normal2,
                               tol_c, cos_cut)
        pka, pkb = pts[ka], pts[kb]
        ball = _ball_cut(pka, pkb, kl, ksh, b_pts, b_shape, b_rad, tiers)
        surf = _surfel_cut(pka, pkb, kl, ksh, f_pts, f_shape, f_normal, tol_c, cos_cut)
        return ball, surf

    # unknown parent edges are part of the optimistic graph without a test
    quotient = connected_components(total, label_p[par_a[~par_cert]], label_p[par_b[~par_cert]])
    label_o = quotient[label_p]
    tick("label_propagation")
    # The short stages (below 2 tol_c, where C2b cannot fire) share one
    # node–ball relation; the last stage holds the long edges.
    short = [f * r_s for f in _STAGE_FRACTIONS if f * r_s < 2.0 * tol_c]
    bounds = short + [float("inf")]
    stage_of = torch.bucketize(cand_len, torch.tensor(short, device=device, dtype=dtype))
    relation = None
    n_tested = n_cut_c1 = n_cut_c2b = n_unknown = 0
    cc_rounds = []
    debug_keep = []
    for stage in range(len(bounds)):
        sel = stage_of == stage
        if stage > 0:
            sel &= label_o[cand_a] != label_o[cand_b]
        k_idx = sel.nonzero().squeeze(1)
        if k_idx.numel() == 0:
            continue
        ka, kb, kl = cand_a[k_idx], cand_b[k_idx], cand_len[k_idx]
        if stage < len(short) and ka.numel() * per > _DENSE_SEGMENT_LIMIT:
            if relation is None:
                relation = _node_ball_relation(s_idx, pts, shape, total, o_pts, o_shape, o_rad,
                                               tiers, 0.5 * short[-1])
            cut_ball = _ball_cut_by_nodes(relation, ka, kb, pts, o_pts, o_rad)
            cut_surf = torch.zeros_like(cut_ball)
        else:
            cut_ball, cut_surf = cut_tests(ka, kb, kl, o_pts, o_shape, o_rad, cut_free2)
        tick("cut_tests")
        ok = ~(cut_ball | cut_surf)
        quotient, rounds = connected_components(
            total, label_o[ka[ok]], label_o[kb[ok]], return_iterations=True
        )
        label_o = quotient[label_o]
        cc_rounds.append(rounds)
        tick("label_propagation")
        n_tested += k_idx.numel()
        n_cut_c1 = n_cut_c1 + cut_ball.sum()
        n_cut_c2b = n_cut_c2b + (cut_surf & ~cut_ball).sum()
        n_unknown = n_unknown + ok.sum()
        if debug:
            debug_keep.append((ka[ok], kb[ok]))
    if debug:
        counts_debug = dict(
            pts=pts, val=val, cert_a=cert_a, cert_b=cert_b, par_a=par_a, par_b=par_b,
            keep_a=torch.cat([k[0] for k in debug_keep]) if debug_keep else cand_a[:0],
            keep_b=torch.cat([k[1] for k in debug_keep]) if debug_keep else cand_a[:0],
            cand_a=cand_a, cand_b=cand_b,
        )
    else:
        counts_debug = None
    minus = torch.full_like(label_o, -1)
    label_o = torch.where(solid, label_o, minus)
    label_p = torch.where(solid, label_p, minus)

    # ---- main body and fragments
    main_o = _main_label(label_o, num_shapes, per)
    # Pessimistic main body: the most populous certified component inside the
    # optimistic main body. (The origin's own ball may overlap no other, so
    # the origin node is not a reliable anchor of the certified graph.)
    main_p = _main_label(torch.where(label_o == main_o[shape], label_p, minus),
                         num_shapes, per, use_origin=False)
    neg_inf = torch.full((total,), float("-inf"), dtype=dtype, device=device)
    size_o = neg_inf.scatter_reduce(0, label_o[solid], val[solid], reduce="amax")
    size_p = neg_inf.scatter_reduce(0, label_p[solid], val[solid], reduce="amax")
    is_root_o = solid & (label_o == index)
    is_root_p = solid & (label_p == index)
    # A fragment needs evidence of solid. Either a confident inside node (size > δ), or -- for a
    # piece thinner than 2δ, which has no such node -- a cluster of at least 6 nodes of which at
    # least 2 are inside beyond the consistency error. A lone band node is within noise of the
    # surface and carries no solid. (A false cluster split off the main surface is dropped by
    # ``confirm``, whose joining segment is then inside everywhere.)
    ones = torch.ones((), dtype=dtype, device=device)
    zeros_n = torch.zeros((total,), dtype=dtype, device=device)
    count_o = zeros_n.scatter_add(0, label_o[solid], ones.expand(int(solid.sum())))
    inside_o = zeros_n.scatter_add(0, label_o[solid], (val[solid] > e_p[solid]).to(dtype))
    # ... or it contains a wall pair (two surface nodes facing away from each other with solid
    # between them), which is direct evidence of a thin solid even when no sample fell inside.
    wall_o = zeros_n.scatter_add(0, label_o[ea[wall_site]],
                                 ones.expand(int(wall_site.sum()))) > 0
    thin_piece = (count_o >= 6) & ((inside_o >= 2) | wall_o)
    frag0 = is_root_o & (index != main_o[shape]) & ((size_o > delta_p) | thin_piece)
    num_pieces = (frag0.view(num_shapes, per).sum(dim=1) + (main_o >= 0).long())
    if full:
        # kind 1: separate only in the certified graph. It must carry a core
        # ball of radius kind1_core, else every barely-deep node whose ball
        # happens to touch no other would be reported.
        frag1 = (
            is_root_p & (index != main_p[shape]) & (label_o == main_o[shape])
            & (size_p - e_p >= kind1_core)
        )
    else:
        frag1 = torch.zeros_like(frag0)
    roots = torch.cat([frag0.nonzero().squeeze(1), frag1.nonzero().squeeze(1)])
    kinds = torch.cat([torch.zeros(int(frag0.sum()), dtype=torch.long, device=device),
                       torch.ones(int(frag1.sum()), dtype=torch.long, device=device)])
    sizes = torch.where(kinds == 0, size_o[roots], size_p[roots])
    keep = _topk_per_group(shape[roots], sizes, max_fragments, largest=True)
    roots, kinds, sizes = roots[keep], kinds[keep], sizes[keep]
    frag_shape = shape[roots]
    lab_o2, lab_p2 = label_o.view(num_shapes, per), label_p.view(num_shapes, per)
    is0 = (kinds == 0)[:, None]
    frag_mask = torch.where(is0, lab_o2[frag_shape] == roots[:, None],
                            lab_p2[frag_shape] == roots[:, None])
    body_mask = torch.where(is0, lab_o2[frag_shape] == main_o[frag_shape, None],
                            lab_p2[frag_shape] == main_p[frag_shape, None])
    tick("findings_components")
    frag_a, frag_b, frag_gap, frag_na, frag_nb = _closest_pairs(pts2, frag_shape, frag_mask, body_mask)
    tick("findings_closest_pairs")
    n_kind1_dropped = 0
    if bool((kinds == 1).any()):
        # A kind-1 fragment is bridged along its joining segment. If that
        # segment is known to cross free space (a slot between two arms of one
        # piece), bridging would create a handle: drop the fragment. Short
        # crossings are mostly below what these tests resolve; `confirm`
        # (graph_losses) repeats the check with field values on the segment.
        k1 = ((kinds == 1) & (frag_na >= 0)).nonzero().squeeze(1)
        na, nb = frag_na[k1], frag_nb[k1]
        u1 = (frag_b[k1] - frag_a[k1]) / frag_gap[k1].clamp_min(1e-12).unsqueeze(-1)
        d_a, d_b = (u1 * normal[na]).sum(dim=-1), (u1 * normal[nb]).sum(dim=-1)
        res1 = gap_res_b[frag_shape[k1]]
        seg_cut = band[na] & (d_a < -cos_one) & (-frag_gap[k1] * d_a > res1)
        seg_cut |= band[nb] & (d_b > cos_one) & (frag_gap[k1] * d_b > res1)
        g_ball, g_surf = _dense_cuts(frag_a[k1], frag_b[k1], frag_gap[k1], frag_shape[k1],
                                     pts2, cut_free2, surfel2, normal2, tol_c, cos_cut)
        seg_cut |= g_ball | g_surf
        keep_rows = torch.ones_like(kinds, dtype=torch.bool)
        keep_rows[k1[seg_cut]] = False
        n_kind1_dropped = seg_cut.sum()
        roots, kinds, sizes, frag_shape = roots[keep_rows], kinds[keep_rows], sizes[keep_rows], frag_shape[keep_rows]
        frag_mask, frag_a, frag_b, frag_gap = frag_mask[keep_rows], frag_a[keep_rows], frag_b[keep_rows], frag_gap[keep_rows]
    tick("findings_kind1_guard")

    # ---- thin sites
    # Thin sites are reported on the main body only. A detached fragment is something to
    # delete or bridge; thickening it would feed it (a small blob looks like a thin wall from
    # every side), and widening a gap inside it is pointless.
    edge_main = main_o[e_shape]
    in_main = (edge_main >= 0) & (label_o[ea] == edge_main) & (label_o[eb] == edge_main)
    gap_site &= in_main
    wall_site &= in_main

    def pick_sites(flag: Tensor) -> Tensor:
        """The ``max_sites`` shortest sites per shape, at most one per grid cell.

        Guesses pile up on sharp convex features (every sample in the outer
        wedge of an edge or corner projects onto it), so without the cell
        rule the shortest sites would be many copies of the same place.
        DEVIATION: "shortest" is taken at the resolution of the band (lengths
        are binned to 2δ) and ties are broken by a fixed hash of the cell, so
        that sites of equal thinness are spread over the feature instead of
        all sitting on its rim.
        """
        idx = flag.nonzero().squeeze(1)
        if idx.numel() == 0:
            return idx
        mid = 0.5 * (pts[ea[idx]] + pts[eb[idx]])
        cell = torch.floor((mid + _COORD_LIMIT) / tol_c).to(torch.long)
        side = int(2 * _COORD_LIMIT / tol_c) + 2
        key = ((e_shape[idx] * side + cell[:, 0]) * side + cell[:, 1]) * side + cell[:, 2]
        order = torch.argsort(s_len[idx], stable=True)
        order = order[torch.argsort(key[order], stable=True)]
        sorted_key = key[order]
        first = torch.ones_like(sorted_key, dtype=torch.bool)
        first[1:] = sorted_key[1:] != sorted_key[:-1]
        idx, cell = idx[order[first]], cell[order[first]]
        mix = (cell[:, 0] * 73856093 + cell[:, 1] * 19349663 + cell[:, 2] * 83492791) % 1000003
        score = torch.round(s_len[idx] / (2.0 * delta_b[e_shape[idx]])) + mix.to(dtype) / 1000003.0
        return idx[_topk_per_group(e_shape[idx], score, max_sites, largest=False)]

    g_idx = pick_sites(gap_site)
    w_idx = pick_sites(wall_site)
    gap_mid = 0.5 * (pts[ea[g_idx]] + pts[eb[g_idx]])
    wall_mid = 0.5 * (pts[ea[w_idx]] + pts[eb[w_idx]])
    tick("findings_sites")

    # ================================================================ outside graph
    # Far-field nodes (|x| ≥ R_b) are one component by definition: they are
    # merged into one representative per shape and far–far pairs are skipped.
    out_idx = outside.nonzero().squeeze(1)
    far = outside & (pts.norm(dim=-1) >= bound_radius)
    rep = torch.full((num_shapes,), total, dtype=torch.long, device=device)
    far_idx = far.nonzero().squeeze(1)
    rep = rep.scatter_reduce(0, shape[far_idx], far_idx, reduce="amin")
    has_far = rep < total
    node_map = torch.where(far, rep.clamp_max(total - 1)[shape], index)
    near_idx = (outside & ~far).nonzero().squeeze(1)
    ja, jb, o_len = pairs_within(
        pts[near_idx], shape[near_idx], pts[out_idx], shape[out_idx], r_v, return_distance=True
    )
    oa, ob = near_idx[ja], out_idx[jb]
    once = far[ob] | (oa < ob)   # near–near pairs come in both orders
    oa, ob, o_len = oa[once], ob[once], o_len[once]
    tick("outside_pairs")
    o_cert = (o_len <= rho_free[oa] + rho_free[ob]) if full else torch.zeros_like(oa, dtype=torch.bool)
    ma, mb = node_map[oa], node_map[ob]
    out_p, it_op = connected_components(total, ma[o_cert], mb[o_cert], return_iterations=True)
    tick("outside_labels")

    b_idx = (cut_in > 0).nonzero().squeeze(1)
    b_pts, b_shape, b_rad = pts[b_idx], shape[b_idx], cut_in[b_idx]
    cut_in2 = cut_in.view(num_shapes, per)
    out_label = out_p
    out_counts = dict(tested=0, cut_ball=0, cut_c2b=0)
    pending = ~o_cert
    stage_len = _OUT_STAGE1_FRACTION * r_v
    for stage in (0, 1):
        sel = pending & (out_label[ma] != out_label[mb])
        sel &= (o_len <= stage_len) if stage == 0 else (o_len > stage_len)
        k_idx = sel.nonzero().squeeze(1)
        if k_idx.numel() == 0:
            continue
        ka, kb, kl = oa[k_idx], ob[k_idx], o_len[k_idx]
        cut_ball, cut_surf = cut_tests(ka, kb, kl, b_pts, b_shape, b_rad, cut_in2)
        ok = ~(cut_ball | cut_surf)
        quotient = connected_components(total, out_label[ma[k_idx][ok]], out_label[mb[k_idx][ok]])
        out_label = quotient[out_label]
        out_counts["tested"] += k_idx.numel()
        out_counts["cut_ball"] = out_counts["cut_ball"] + cut_ball.sum()
        out_counts["cut_c2b"] = out_counts["cut_c2b"] + (cut_surf & ~cut_ball).sum()
    out_label = torch.where(outside, out_label[node_map], minus)
    tick("outside_cuts")

    # ---- cavities: optimistic outside components with no far-field node
    out2 = out_label.view(num_shapes, per)
    out_count = torch.bincount(out_label[outside], minlength=total)
    biggest = out_count.view(num_shapes, per).argmax(dim=1) + torch.arange(num_shapes, device=device) * per
    far_label = torch.where(has_far, out_label[rep.clamp_max(total - 1)], biggest)
    far_label = torch.where(out_count.view(num_shapes, per).sum(dim=1) > 0, far_label,
                            torch.full_like(far_label, -1))
    cav_root = outside & (out_label == index) & (index != far_label[shape])
    num_cavities = cav_root.view(num_shapes, per).sum(dim=1)
    c_roots = cav_root.nonzero().squeeze(1)
    c_roots = c_roots[_topk_per_group(shape[c_roots], out_count[c_roots].to(dtype),
                                      max_cavities, largest=True)]
    cav_shape = shape[c_roots]
    cav_mask = out2[cav_shape] == c_roots[:, None]
    far_mask = out2[cav_shape] == far_label[cav_shape, None]
    tick("findings_components")
    cav_a, cav_b, _, _, _ = _closest_pairs(pts2, cav_shape, cav_mask, far_mask)
    tick("findings_closest_pairs")

    # ---- statistics (one transfer for all tensor counts)
    counts.update(
        nodes_inside=inside.sum(), nodes_band=band.sum(), nodes_outside=outside.sum(),
        nodes_solid=solid.sum(), nodes_deep=deep.sum(), nodes_surfel=surfel.sum(),
        surface_edges=n_primary, surface_rescue_edges=ea.numel() - n_primary,
        surface_certified=s_cert.sum(),
        surface_cut_c2a=c2a.sum(), surface_cut_local=local_cut.sum(),
        surface_candidates=n_surface_cand,
        volume_certified=deep_out["n_vol_cert"], volume_dropped_hidden_gap=deep_out["n_vol_dropped"],
        volume_candidates=deep_out["cand_a"].numel(),
        deep_certified_edges=deep_out["cert_a"].numel(), deep_max_per_shape=deep_out["n_deep_max"],
        parent_certified=par_cert.sum(), parent_unknown=(~par_cert).sum(),
        tested=n_tested, tested_cut_c1=n_cut_c1, tested_cut_c2b=n_cut_c2b,
        tested_unknown=n_unknown, untested_redundant=cand_len.numel() - n_tested,
        outside_edges=oa.numel(), outside_certified=o_cert.sum(),
        outside_tested=out_counts["tested"], outside_cut_ball=out_counts["cut_ball"],
        outside_cut_c2b=out_counts["cut_c2b"],
        fragments_kind0=(kinds == 0).sum(), fragments_kind1=(kinds == 1).sum(),
        fragments_kind1_dropped_cut_segment=n_kind1_dropped,
        cavities=c_roots.numel(), gap_sites=g_idx.numel(), wall_sites=w_idx.numel(),
        cc_rounds_pess=it_p, cc_rounds_stages=sum(cc_rounds), cut_stages_run=len(cc_rounds),
    )
    names = list(counts)
    packed = torch.stack([torch.as_tensor(counts[k], device=device).to(torch.long).reshape(())
                          for k in names]).tolist()
    stats: Dict[str, object] = dict(zip(names, packed))
    stats.update(mode=mode, r_s=r_s, r_v=r_v, tol_c=tol_c,
                 delta=float(delta_b.mean()), e=float(e_b.mean()),
                 gap_resolution=float(gap_res_b.mean()))
    tick("stats")
    if profile:
        stats["timings_ms"] = dict(timings)
    if counts_debug is not None:
        stats["debug"] = counts_debug
    stats["total_ms"] = (time.perf_counter() - t_start) * 1e3

    def view(v: Tensor) -> Tensor:
        return v.view(num_shapes, per)

    return ShapeGraph(
        inside=view(inside), band=view(band), outside=view(outside),
        solid_label=view(label_o), solid_label_pess=view(label_p), outside_label=out2,
        main_label=main_o,
        frag_shape=frag_shape, frag_kind=kinds, frag_size=sizes, frag_gap=frag_gap,
        frag_a=frag_a, frag_b=frag_b, frag_mask=frag_mask,
        cav_shape=cav_shape, cav_a=cav_a, cav_b=cav_b, cav_mask=cav_mask,
        gap_shape=e_shape[g_idx], gap_mid=gap_mid, gap_len=s_len[g_idx],
        wall_shape=e_shape[w_idx], wall_mid=wall_mid, wall_len=s_len[w_idx],
        stats=stats, num_pieces=num_pieces, num_cavities=num_cavities,
        main_label_pess=main_p, far_label=far_label, delta=delta_b, mode=mode,
    )
