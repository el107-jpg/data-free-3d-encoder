"""Losses that act on the findings of the point graph.

The graph (``graph.py``) only selects points. Gradients enter here, through
ordinary field evaluations at those points:

- ``confirm``: no-grad check of the joining segments, guarding the destructive
  actions (delete, fill) and the bridging of kind-1 fragments;
- ``action_losses``: bridge, delete, fill, widen, thicken;
- ``hard_negatives``: points worth keeping in the trainer's anchor cache.

Field access: every function takes ``(decoder, mod)``. ``decoder`` may be a
``ModulatedSphereDecoder`` with its prepared modulation ``mod``, or (with
``mod=None``) any callable ``field_fn(shape_index Q, points Q×3) -> values Q``,
which is what the tests use with analytic fields.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Dict, Optional

import torch
from torch import Tensor

from .graph import ShapeGraph, _cfg

FieldFn = Callable[[Tensor, Tensor], Tensor]
ACTION_KEYS = ("bridge", "delete", "fill", "widen", "thicken")
# kind-1 joining segments are sampled this many times denser in `confirm`
_KIND1_DENSITY = 4


class DecoderField:
    """Adapter: evaluate a modulated decoder at loose points tagged with a shape index.

    Points are packed into one ``B×L×3`` block (``L`` = most points of any
    shape, padded with the origin) and queried in a single decoder call.
    """

    def __init__(self, decoder, mod, num_shapes: int) -> None:
        self.decoder, self.mod, self.num_shapes = decoder, mod, int(num_shapes)

    def __call__(self, shape_index: Tensor, points: Tensor) -> Tensor:
        if points.shape[0] == 0:
            return points.new_zeros(0)
        order = torch.argsort(shape_index, stable=True)
        sh = shape_index[order]
        counts = torch.bincount(sh, minlength=self.num_shapes)
        rank = torch.arange(sh.numel(), device=sh.device) - (torch.cumsum(counts, 0) - counts)[sh]
        block = points.new_zeros(self.num_shapes, int(counts.max()), 3)
        block[sh, rank] = points[order]
        values = self.decoder.query(self.mod, block)
        out = values.new_zeros(points.shape[0])
        out[order] = values[sh, rank]
        return out


def as_field_fn(decoder, mod, num_shapes: int) -> FieldFn:
    """``(decoder, mod)`` or a plain callable (``mod is None``) as a field function."""
    if mod is None and not hasattr(decoder, "query"):
        return decoder
    return DecoderField(decoder, mod, num_shapes)


def segment_points(a: Tensor, b: Tensor, samples: int) -> Tensor:
    """``R×S×3`` points at the centres of ``S`` equal pieces of each segment."""
    t = (torch.arange(samples, device=a.device, dtype=a.dtype) + 0.5) / samples
    return a[:, None, :] + (b - a)[:, None, :] * t[None, :, None]


def _delta(graph: ShapeGraph, cfg) -> Tensor:
    if graph.delta is not None:
        return graph.delta
    value = float(graph.stats.get("delta", _cfg(cfg, "delta_min", 0.01)))
    return torch.full((graph.num_shapes,), value, device=graph.inside.device)


@torch.no_grad()
def confirm(decoder, mod, graph: ShapeGraph, cfg) -> ShapeGraph:
    """Re-check every joining segment with field values and drop what fails.

    With ``δ`` the band half-width the graph used:

    - a kind-0 fragment whose segment is inside everywhere (``f ≥ −δ`` at all
      samples) was connected after all: dropped, so it is neither deleted nor
      bridged;
    - a cavity whose segment is outside everywhere (``f ≤ δ``) is open to the
      far field: dropped, so it is not filled;
    - a kind-1 fragment whose segment has a confidently outside sample
      (``f < −δ``) is separated from the main body by a real gap there, and
      both belong to one piece: bridging would create a handle, so it is
      dropped. (Sampled ``4×`` denser, because this segment can be long.)

    Rows without a joining segment (NaN ends) are kept unchanged.
    """
    device = graph.inside.device
    samples = int(_cfg(cfg, "segment_samples", 8))
    field_fn = as_field_fn(decoder, mod, graph.num_shapes)
    delta = _delta(graph, cfg)
    n_frag, n_cav = graph.frag_shape.numel(), graph.cav_shape.numel()
    if n_frag + n_cav == 0:
        return graph

    dense = samples * _KIND1_DENSITY
    frag_pts = segment_points(graph.frag_a, graph.frag_b, dense)            # F×4S×3
    cav_pts = segment_points(graph.cav_a, graph.cav_b, samples)             # K×S×3
    points = torch.cat([frag_pts.reshape(-1, 3), cav_pts.reshape(-1, 3)])
    shape = torch.cat([graph.frag_shape.repeat_interleave(dense),
                       graph.cav_shape.repeat_interleave(samples)])
    finite = torch.isfinite(points).all(dim=-1)
    values = torch.full((points.shape[0],), float("nan"), device=device, dtype=points.dtype)
    if bool(finite.any()):
        values[finite] = field_fn(shape[finite], points[finite]).detach().to(points.dtype)
    f_frag = values[: n_frag * dense].view(n_frag, dense)
    f_cav = values[n_frag * dense:].view(n_cav, samples)

    d_frag = delta[graph.frag_shape][:, None]
    has_seg = torch.isfinite(f_frag).all(dim=1)
    all_inside = (f_frag >= -d_frag).all(dim=1)
    any_outside = (f_frag < -d_frag).any(dim=1)
    kind0 = graph.frag_kind == 0
    drop_frag = has_seg & torch.where(kind0, all_inside, any_outside)
    d_cav = delta[graph.cav_shape][:, None]
    drop_cav = torch.isfinite(f_cav).all(dim=1) & (f_cav <= d_cav).all(dim=1)

    keep_f, keep_c = ~drop_frag, ~drop_cav
    num_pieces, num_cavities = graph.num_pieces, graph.num_cavities
    if num_pieces is not None:
        gone = torch.bincount(graph.frag_shape[drop_frag & kind0], minlength=graph.num_shapes)
        num_pieces = num_pieces - gone
    if num_cavities is not None:
        num_cavities = num_cavities - torch.bincount(graph.cav_shape[drop_cav],
                                                     minlength=graph.num_shapes)
    stats = dict(graph.stats)
    stats.update(
        confirm_dropped_kind0=int((drop_frag & kind0).sum()),
        confirm_dropped_kind1=int((drop_frag & ~kind0).sum()),
        confirm_dropped_cavities=int(drop_cav.sum()),
    )
    return dataclasses.replace(
        graph,
        frag_shape=graph.frag_shape[keep_f], frag_kind=graph.frag_kind[keep_f],
        frag_size=graph.frag_size[keep_f], frag_gap=graph.frag_gap[keep_f],
        frag_a=graph.frag_a[keep_f], frag_b=graph.frag_b[keep_f],
        frag_mask=graph.frag_mask[keep_f],
        cav_shape=graph.cav_shape[keep_c], cav_a=graph.cav_a[keep_c],
        cav_b=graph.cav_b[keep_c], cav_mask=graph.cav_mask[keep_c],
        stats=stats, num_pieces=num_pieces, num_cavities=num_cavities,
    )


def fragment_actions(graph: ShapeGraph, cfg) -> Tensor:
    """Per fragment: True = delete, False = bridge.

    A kind-0 fragment is deleted when it is small (``size < delete_size``) or
    far (``gap > delete_gap``), else bridged. Kind-1 fragments are always
    bridged. A fragment with no joining segment can only be deleted.
    """
    t = float(_cfg(cfg, "min_feature", 0.08))
    delete_size = float(_cfg(cfg, "delete_size", 0.5 * t))
    delete_gap = float(_cfg(cfg, "delete_gap", 2.0 * t))
    kind0 = graph.frag_kind == 0
    no_segment = ~torch.isfinite(graph.frag_a).all(dim=-1)
    return kind0 & ((graph.frag_size < delete_size) | (graph.frag_gap > delete_gap) | no_segment)


def _per_shape_mean(values: Tensor, shape: Tensor, weight: Tensor, num_shapes: int) -> Tensor:
    """Sum over shapes of the weighted mean of ``values`` within each shape, over ``B``."""
    total = values.new_zeros(num_shapes).index_add(0, shape, values * weight)
    count = values.new_zeros(num_shapes).index_add(0, shape, weight)
    return (total / count.clamp_min(1.0)).sum() / num_shapes


def action_losses(decoder, mod, fb, graph: ShapeGraph, cfg) -> Dict[str, Tensor]:
    """The five action losses, each a scalar (zero when nothing was found).

    - ``bridge``: ``relu(−f)²`` at ``segment_samples`` points on the joining
      segment of every fragment that is bridged (new field evaluations);
    - ``delete``: ``relu(F)²`` over the member points of every fragment that
      is deleted, with the values already in ``fb``;
    - ``fill``: ``relu(−F)²`` over the member points of every cavity, from ``fb``;
    - ``widen``: ``relu(f + t/2)²`` at gap-site midpoints where ``f < 0``;
    - ``thicken``: ``relu(t/2 − f)²`` at wall-site midpoints where ``f > 0``.

    Normalisation: each fragment / cavity contributes the mean over its
    points; sites contribute their mean per shape; everything is summed and
    divided by the number of shapes ``B``. So each loss reads as "squared
    violation per shape" and does not grow with the number of sample points.

    Call ``confirm`` first: it removes the findings on which these actions
    would be wrong.
    """
    num_shapes = graph.num_shapes
    values = torch.cat([fb.f, fb.fp], dim=1)                     # B×2N, in the autograd graph
    dtype, device = values.dtype, values.device
    zero = values.new_zeros(())
    out = {key: zero for key in ACTION_KEYS}
    samples = int(_cfg(cfg, "segment_samples", 8))
    t = float(_cfg(cfg, "min_feature", 0.08))
    field_fn = as_field_fn(decoder, mod, num_shapes)

    delete = fragment_actions(graph, cfg)
    bridge = ~delete

    # ---- losses on values that already exist
    if bool(delete.any()):
        mask = graph.frag_mask[delete].to(dtype)                 # D×2N
        member = values[graph.frag_shape[delete]]
        per = (torch.relu(member).square() * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        out["delete"] = per.sum() / num_shapes
    if graph.cav_shape.numel() > 0:
        mask = graph.cav_mask.to(dtype)
        member = values[graph.cav_shape]
        per = (torch.relu(-member).square() * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        out["fill"] = per.sum() / num_shapes

    # ---- losses that need new evaluations: one field call for all of them
    seg = segment_points(graph.frag_a[bridge], graph.frag_b[bridge], samples).to(dtype)
    n_bridge = seg.shape[0] * samples
    n_gap, n_wall = graph.gap_shape.numel(), graph.wall_shape.numel()
    if n_bridge + n_gap + n_wall == 0:
        return out
    points = torch.cat([seg.reshape(-1, 3), graph.gap_mid.to(dtype), graph.wall_mid.to(dtype)])
    shape = torch.cat([graph.frag_shape[bridge].repeat_interleave(samples),
                       graph.gap_shape, graph.wall_shape])
    f_new = field_fn(shape, points.detach())
    f_bridge = f_new[:n_bridge]
    f_gap = f_new[n_bridge:n_bridge + n_gap]
    f_wall = f_new[n_bridge + n_gap:]
    if n_bridge > 0:
        per = torch.relu(-f_bridge).square().view(-1, samples).mean(dim=1)
        out["bridge"] = per.sum() / num_shapes
    # A real gap of width L reads about -L/2 at its midpoint, a real wall about +L/2. A site
    # whose midpoint sits near the surface instead (|f| below a quarter of L) is a false
    # alarm, e.g. two nearby points on the same face; acting on it would push the surface
    # itself in or out by t/2 and can never be satisfied.
    if n_gap > 0:
        active = (f_gap.detach() < -0.25 * graph.gap_len.to(dtype)).to(dtype)    # really free space
        out["widen"] = _per_shape_mean(torch.relu(f_gap + 0.5 * t).square(), graph.gap_shape,
                                       active, num_shapes)
    if n_wall > 0:
        active = (f_wall.detach() > 0.25 * graph.wall_len.to(dtype)).to(dtype)   # really solid
        out["thicken"] = _per_shape_mean(torch.relu(0.5 * t - f_wall).square(), graph.wall_shape,
                                         active, num_shapes)
    return out


@torch.no_grad()
def hard_negatives(graph: ShapeGraph, fb, cfg, per_shape: int,
                   generator: Optional[torch.Generator] = None) -> Tensor:
    """``B×per_shape×3`` points (NaN-padded) to keep for later steps on the same codes.

    Per shape, in this order of preference: thin-site midpoints and the
    sample points on joining segments (where the actions act), then a random
    subset of the member points of fragments and cavities.
    """
    num_shapes = graph.num_shapes
    device = graph.inside.device
    pts = torch.cat([fb.x, fb.xp], dim=1).detach()               # B×2N×3
    dtype = pts.dtype
    samples = int(_cfg(cfg, "segment_samples", 8))
    out = torch.full((num_shapes, per_shape, 3), float("nan"), dtype=dtype, device=device)
    if per_shape <= 0:
        return out

    first_pts = [graph.gap_mid.to(dtype), graph.wall_mid.to(dtype),
                 segment_points(graph.frag_a, graph.frag_b, samples).reshape(-1, 3).to(dtype),
                 segment_points(graph.cav_a, graph.cav_b, samples).reshape(-1, 3).to(dtype)]
    first_shape = [graph.gap_shape, graph.wall_shape,
                   graph.frag_shape.repeat_interleave(samples),
                   graph.cav_shape.repeat_interleave(samples)]
    fi = graph.frag_mask.nonzero(as_tuple=True)
    ci = graph.cav_mask.nonzero(as_tuple=True)
    f_shape, c_shape = graph.frag_shape[fi[0]], graph.cav_shape[ci[0]]
    second_pts = [pts[f_shape, fi[1]], pts[c_shape, ci[1]]]
    second_shape = [f_shape, c_shape]

    points = torch.cat(first_pts + second_pts)
    shape = torch.cat(first_shape + second_shape)
    n_first = sum(p.shape[0] for p in first_pts)
    if points.shape[0] == 0:
        return out
    rand = torch.rand(points.shape[0], generator=generator,
                      device=generator.device if generator is not None else device).to(device)
    score = rand + (torch.arange(points.shape[0], device=device) >= n_first).to(rand.dtype)
    ok = torch.isfinite(points).all(dim=-1)
    points, shape, score = points[ok], shape[ok], score[ok]
    order = torch.argsort(score)
    order = order[torch.argsort(shape[order], stable=True)]
    sh = shape[order]
    counts = torch.bincount(sh, minlength=num_shapes)
    rank = torch.arange(sh.numel(), device=device) - (torch.cumsum(counts, 0) - counts)[sh]
    keep = rank < per_shape
    out[sh[keep], rank[keep]] = points[order][keep]
    return out
