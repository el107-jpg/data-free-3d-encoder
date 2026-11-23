"""Loss terms for the data-free latent space.

Sign convention: the field is positive inside the shape, so the spatial
gradient ``g`` points into the solid.  Shapes live in the box ``[-1, 1]^3``.

Every function takes a :class:`~arm_planner.data_free.field.FieldBatch`.
Constraint terms are non-negative and zero on a field that satisfies them.
Nothing here evaluates the network except :func:`grad_change`.

Row 0 of the volume samples is the origin (``fb.x[:, 0] == 0``).  The base
field ``R - |x|`` has a kink there (zero gradient, so the one-step guess does
not move), so the origin row is left out of the Eikonal, consistency and
gradient-change terms.  It is used by every other term.
"""

from __future__ import annotations

import functools
import math
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from .field import FieldBatch, field_and_grad
from .latent import chord

__all__ = [
    "BOX_VOLUME",
    "DESCRIPTOR_NAMES",
    "DESCRIPTOR_SCALES",
    "eikonal",
    "consistency",
    "consistency_error",
    "grad_change",
    "crease",
    "barrier",
    "origin_anchor",
    "soft_inside",
    "min_volume",
    "moments",
    "pose",
    "pose_terms",
    "POSE_KEYS",
    "soft_ball_trace",
    "soft_ball_moment",
    "pose_moment_target",
    "pose_size_target",
    "extent_pose",
    "descriptors",
    "diversity",
]

BOX_VOLUME = 8.0

DESCRIPTOR_NAMES: Tuple[str, ...] = (
    "volume_fraction",
    "surface_area",
    "sphericity",
    "elongation",
    "flatness",
    "mean_depth",
    "offset_growth",
    "depth_p25",
    "depth_p50",
    "depth_p90",
)

# Levels of the thickness profile (fractions of the inside points, from the shallow end).
_DEPTH_LEVELS: Tuple[float, ...] = (0.25, 0.5, 0.9)

# Fixed scale of each descriptor: the size of a difference that counts as one
# unit of shape distance in phase 2.  Same order as ``DESCRIPTOR_NAMES``.
DESCRIPTOR_SCALES: Tuple[float, ...] = (0.05, 3.0, 0.25, 0.5, 0.5, 0.1, 0.25, 0.05, 0.08, 0.12)

_NORM_EPS = 1e-12
_WEIGHT_EPS = 1e-6
_EIG_FLOOR = 1e-8
_AREA_EPS = 1e-4
_POW_EPS = 1e-9
_TARGET_FLOOR = 1e-6


def _norm(v: Tensor) -> Tensor:
    """Vector length along the last axis with a finite gradient at zero."""
    return torch.sqrt(v.square().sum(dim=-1) + _NORM_EPS)


def _rand(shape, generator: Optional[torch.Generator], device, normal: bool = False) -> Tensor:
    fn = torch.randn if normal else torch.rand
    if generator is None:
        return fn(shape, device=device)
    return fn(shape, generator=generator, device=generator.device).to(device)


# --------------------------------------------------------------------- SDF terms
def eikonal(fb: FieldBatch) -> Tensor:
    """Mean ``(|g| - 1)^2`` over the volume samples (origin row excluded).

    With ``fb.fp_weight`` (set when starting points near the surface are left out of the
    consistency term) the same points are left out here, so the term works on the field around
    the surface and not on the surface itself.
    """
    error = (_norm(fb.g[:, 1:]) - 1.0).square()
    weight = getattr(fb, "fp_weight", None)
    if weight is None:
        return error.mean()
    weight = weight[:, 1:]
    return (error * weight).sum() / weight.sum().clamp_min(1.0)


def consistency(fb: FieldBatch) -> Tensor:
    """Mean ``|f(x - f g)|`` over the one-step surface guesses (origin row excluded).

    With ``fb.fp_weight`` it is the weighted mean (used to leave out starting points near the surface).
    """
    miss = fb.fp[:, 1:].abs()
    weight = getattr(fb, "fp_weight", None)
    if weight is None:
        return miss.mean()
    weight = weight[:, 1:]
    return (miss * weight).sum() / weight.sum().clamp_min(1.0)


def consistency_error(fb: FieldBatch) -> Tensor:
    """Detached median of ``|fp|`` over the whole batch: the graph's ``e``."""
    with torch.no_grad():
        return fb.fp[:, 1:].abs().reshape(-1).median()


def code_isotropy(decoder, codes: Tensor, count: int, generator: Optional[torch.Generator] = None) -> Tuple[Tensor, Tensor]:
    """How unevenly the code's directions reach the main network; zero when they all count the same.

    For every code, ``count`` random unit directions along the sphere are pushed through the side
    network (code -> scales and shifts of the main network) with a forward-mode derivative. The
    term is the variance over those directions of ``log |change of the scales and shifts|^2``. It
    is zero exactly when a step of the same size in any direction changes the scales and shifts
    by the same amount, and it does not care how large that amount is. Only the side network is
    evaluated, so the cost is negligible. Returns the term and, detached, the typical ratio between
    a strong and a weak direction (``exp`` of the standard deviation of ``log |change|``).
    """
    z = torch.nn.functional.normalize(codes.detach(), dim=-1)
    rows, dim = z.shape
    v = torch.randn(rows, count, dim, generator=generator).to(z)
    v = v - (v * z[:, None]).sum(-1, keepdim=True) * z[:, None]
    v = torch.nn.functional.normalize(v, dim=-1)

    def side(c: Tensor) -> Tensor:
        return decoder.film(decoder.mapping(torch.nn.functional.normalize(c, dim=-1)))

    start = z[:, None].expand(rows, count, dim).reshape(rows * count, dim)
    _, change = torch.func.jvp(side, (start,), (v.reshape(rows * count, dim),))
    size = change.square().sum(-1).clamp_min(1e-20).log().reshape(rows, count)
    term = size.var(dim=1).mean()
    return term, torch.exp(0.5 * size.detach().std(dim=1).mean())


def grad_change(
    decoder,
    fb: FieldBatch,
    cfg,
    generator: Optional[torch.Generator] = None,
    mod=None,
) -> Tensor:
    """Mean ``1 - cos(g(x), g(x + step * v))`` on a random subset of samples.

    ``mod`` is the decoder's prepared modulation for ``fb.codes``; when not
    given, ``fb.mod`` is used, and failing that it is computed here.  This is the one loss that evaluates the
    network (``grad_change_points`` extra points per code).
    """
    batch, count, _ = fb.x.shape
    device = fb.x.device
    subset = max(1, min(int(cfg.grad_change_points), count - 1))
    index = 1 + (_rand((batch, subset), generator, device) * (count - 1)).long().clamp_(max=count - 2)
    direction = _rand((batch, subset, 3), generator, device, normal=True)
    direction = direction / _norm(direction).unsqueeze(-1)
    gather = index.unsqueeze(-1).expand(-1, -1, 3)
    x0 = torch.gather(fb.x, 1, gather).detach()
    g0 = torch.gather(fb.g, 1, gather)
    x1 = x0 + float(cfg.grad_change_step) * direction
    if mod is None:
        mod = getattr(fb, "mod", None)
    if mod is None:
        mod = decoder.prepare(fb.codes)
    _, g1 = field_and_grad(decoder, mod, x1, create_graph=True)
    cosine = (g0 * g1).sum(dim=-1) / (_norm(g0) * _norm(g1))
    return (1.0 - cosine).clamp_min(0.0).mean()


def crease(
    decoder,
    fb: FieldBatch,
    cfg,
    generator: Optional[torch.Generator] = None,
    rows: Optional[int] = None,
    mod=None,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Sparse-curvature term: flat surface is free, and one sharp crease costs less than a long bend.

    For each of the first ``rows`` codes, the ``crease_points`` samples closest to the surface are
    taken at their surface guesses ``p``; each gets a partner ``q = p + crease_step * t`` along a
    random direction ``t`` in the surface. With ``n`` the unit normal of the field,
    ``turn = |n(p) - n(q)| / crease_step`` is how fast the surface bends there (1 / radius on a
    sphere, 0 on a plane) and the cost of the pair is ``turn^2 / (turn^2 + crease_scale^2)``:

    * 0 on flat surface, and it rises with the bend as long as the bend is gentle;
    * it levels off at 1, so a pair that straddles a sharp edge costs 1 however sharp the edge is.

    Because the cost levels off, turning the surface by a given angle is cheapest when the turn is
    packed into the narrowest strip: a cube costs a few per cent (only pairs across an edge pay), a
    ball of the same size about ten times more. A plain penalty on the bend (or on its square) does
    the opposite or is indifferent. Pairs count only when both points are within ``crease_band`` of
    the surface and inside the bounding radius. The term is the mean cost of the pairs that count;
    it evaluates the network at ``2 * crease_points`` extra points per code with second derivatives.

    Returns the term and, detached: ``used`` (share of pairs that count), ``turn`` (median bend),
    ``flat`` (share of counted pairs with cost under 0.1) and ``sharp`` (share with cost over 0.9).
    """
    batch, count, _ = fb.x.shape
    rows = batch if rows is None else max(0, min(int(rows), batch))
    zero = fb.f.new_zeros(())
    if rows == 0 or count < 2:
        return zero, {key: zero for key in ("used", "turn", "flat", "sharp")}
    take = max(1, min(int(cfg.crease_points), count - 1))
    step, scale = float(cfg.crease_step), float(cfg.crease_scale)
    with torch.no_grad():
        index = 1 + fb.f[:rows, 1:].abs().topk(take, dim=1, largest=False).indices
        gather = index.unsqueeze(-1).expand(-1, -1, 3)
        p = torch.gather(fb.xp[:rows], 1, gather)
        normal = torch.gather(fb.g[:rows], 1, gather)
        normal = normal / _norm(normal).unsqueeze(-1)
        along = _rand((rows, take, 3), generator, p.device, normal=True).to(p.dtype)
        along = along - (along * normal).sum(dim=-1, keepdim=True) * normal
        along = along / _norm(along).unsqueeze(-1)
        pair = torch.cat([p, p + step * along], dim=1)
    if mod is None:
        mod = getattr(fb, "mod", None)
    if mod is None:
        mod = decoder.prepare(fb.codes)
    if rows < batch:
        mod = mod.index(torch.arange(rows, device=p.device))
    f, g = field_and_grad(decoder, mod, pair, create_graph=True)
    n = g / _norm(g).unsqueeze(-1)
    turn_sq = (n[:, :take] - n[:, take:]).square().sum(dim=-1) / step ** 2
    cost = turn_sq / (turn_sq + scale ** 2)
    with torch.no_grad():
        on = (f.abs() < float(cfg.crease_band)) & (pair.norm(dim=-1) < float(cfg.bound_radius))
        weight = (on[:, :take] & on[:, take:]).to(cost.dtype)
        total = weight.sum().clamp_min(1.0)
        counted = weight > 0
        info = {
            "used": weight.mean(),
            "turn": turn_sq[counted].sqrt().median() if bool(counted.any()) else zero,
            "flat": ((cost < 0.1).to(cost.dtype) * weight).sum() / total,
            "sharp": ((cost > 0.9).to(cost.dtype) * weight).sum() / total,
        }
    return (cost * weight).sum() / total, info


def barrier(fb: FieldBatch, cfg) -> Tensor:
    """Mean of ``relu(barrier_floor - |g|)^2`` over the samples with ``|f| < t/2``.

    The average is over the near-surface samples only (zero when there are
    none), so a local collapse of the gradient is not diluted by the many
    samples far from the surface: the value reads as the typical squared
    shortfall of ``|g|`` in the band.
    """
    near = (fb.f.detach().abs() < 0.5 * float(cfg.min_feature)).to(fb.f.dtype)
    shortfall = torch.relu(float(cfg.barrier_floor) - _norm(fb.g))
    return (near * shortfall.square()).sum() / near.sum().clamp_min(1.0)


# ---------------------------------------------------------------------- validity
def origin_anchor(fb: FieldBatch, cfg) -> Tensor:
    """``relu(origin_margin - f(origin))^2``, mean over codes."""
    return torch.relu(float(cfg.origin_margin) - fb.f[:, 0]).square().mean()


def soft_inside(fb: FieldBatch, cfg) -> Tensor:
    """Soft inside indicator ``sigmoid(f / soft_tau)``, ``B x N``."""
    return torch.sigmoid(fb.f / float(cfg.soft_tau))


def min_volume(fb: FieldBatch, cfg) -> Tensor:
    """``relu(min_volume_fraction - soft volume fraction)^2``, mean over codes."""
    fraction = soft_inside(fb, cfg).mean(dim=1)
    return torch.relu(float(cfg.min_volume_fraction) - fraction).square().mean()


def moments(fb: FieldBatch, cfg) -> Tuple[Tensor, Tensor, Tensor]:
    """Soft volume fraction ``B``, centroid ``B x 3`` and covariance ``B x 3 x 3``.

    Weights are the soft inside indicator over the uniformly sampled volume
    points.  The total weight has a small constant added, so an empty shape
    gives a zero centroid and covariance with finite gradients.
    """
    weight = soft_inside(fb, cfg)
    x = fb.x.detach()
    fraction = weight.mean(dim=1)
    total = weight.sum(dim=1) + _WEIGHT_EPS
    centroid = (weight.unsqueeze(-1) * x).sum(dim=1) / total.unsqueeze(-1)
    offset = x - centroid.unsqueeze(1)
    weighted = weight.unsqueeze(-1) * offset
    cov = torch.einsum("bni,bnj->bij", weighted, offset) / total.reshape(-1, 1, 1)
    return fraction, centroid, cov


def soft_ball_trace(radius: float, tau: float) -> float:
    """Covariance trace of a ball of ``radius`` under the soft indicator ``sigmoid(f / tau)``.

    For ``f = radius - |x|`` the weighted second moment is
    ``int r^4 s(r) dr / int r^2 s(r) dr`` with ``s = sigmoid((radius - r) / tau)``.
    The Sommerfeld expansion gives, up to terms of order ``exp(-radius / tau)``,
    ``(3 R^2 / 5) (1 + 10 q / 3 + 7 q^2 / 3) / (1 + q)`` with ``q = (pi tau / R)^2``.
    The hard-ball value ``3 R^2 / 5`` is the limit ``tau -> 0``; at ``R = 0.5``,
    ``tau = 0.02`` the soft value is 3.7% larger.
    """
    q = (math.pi * float(tau) / float(radius)) ** 2
    return 3.0 * float(radius) ** 2 / 5.0 * (1.0 + 10.0 * q / 3.0 + 7.0 * q * q / 3.0) / (1.0 + q)


def _cov_trace_target(cfg) -> float:
    target = getattr(cfg, "cov_trace_target", None)
    if target is None:
        target = soft_ball_trace(cfg.base_radius, cfg.soft_tau)
    return float(target)


@functools.lru_cache(maxsize=64)
def soft_ball_moment(radius: float, tau: float, power: float) -> float:
    """``power``-th moment of the distance from the centre over a ball, under the soft indicator.

    ``int r^(p+2) s(r) dr / int r^2 s(r) dr`` with ``s = sigmoid((radius - r) / tau)``, by quadrature.
    At ``power = 2`` this is ``soft_ball_trace``; the hard-ball value is ``3 R^p / (3 + p)``.
    """
    steps = 40001
    top = float(radius) + 40.0 * float(tau)
    total_p = total_2 = 0.0
    for k in range(steps):
        r = top * k / (steps - 1)
        z = (float(radius) - r) / float(tau)
        s = 1.0 / (1.0 + math.exp(-z)) if z > -700 else 0.0
        edge = 0.5 if k in (0, steps - 1) else 1.0
        total_p += edge * r ** (float(power) + 2.0) * s
        total_2 += edge * r * r * s
    return total_p / total_2


def pose_moment_target(cfg) -> float:
    """Target of the ``pose_power``-th moment about the origin (the soft ball of radius R by default)."""
    target = getattr(cfg, "pose_moment_target", None)
    if target is None:
        target = soft_ball_moment(float(cfg.base_radius), float(cfg.soft_tau), float(cfg.pose_power))
    return float(target)


def pose_size_target(cfg) -> float:
    """Target size as a length: ``moment^(1/p)``, or ``sqrt(cov_trace_target)`` for the mass-based rule."""
    power = float(getattr(cfg, "pose_power", 2.0))
    if power == 2.0:
        return math.sqrt(_cov_trace_target(cfg))
    return pose_moment_target(cfg) ** (1.0 / power)


def extent_pose(weight: Tensor, x: Tensor, power: float) -> Tuple[Tensor, Tensor]:
    """How far the ``power``-centre is from the origin, and the ``power``-th moment about the origin.

    The centre is the point ``c`` minimising ``sum w |x - c|^p``. Its offset is given to first order by
    one Newton step from the origin, ``H^-1 g`` with ``g = sum w |x|^(p-2) x`` and
    ``H = sum w |x|^(p-2) (I + (p - 2) x x^T / |x|^2)``; ``H`` only sets the units (it is not
    differentiated), so driving the offset to zero drives ``g`` to zero. Returns ``B x 3`` and ``B``.
    At ``power = 2`` these are the mass centre and the second moment about the origin.
    """
    total = weight.sum(dim=1) + _WEIGHT_EPS
    rho = x.norm(dim=-1).clamp_min(1e-9)
    q = weight * rho.pow(float(power) - 2.0)
    g = (q.unsqueeze(-1) * x).sum(dim=1)
    with torch.no_grad():
        unit = x / rho.unsqueeze(-1)
        eye = torch.eye(3, dtype=x.dtype, device=x.device)
        hessian = (q.sum(dim=1) + _WEIGHT_EPS).reshape(-1, 1, 1) * eye + (float(power) - 2.0) * torch.einsum(
            "bn,bni,bnj->bij", q, unit, unit)
    offset = torch.linalg.solve(hessian, g.unsqueeze(-1)).squeeze(-1)
    moment = (weight * rho.pow(float(power))).sum(dim=1) / total
    return offset, moment


def pose(fb: FieldBatch, cfg) -> Tensor:
    """Pin translation, size and principal axes; mean over codes.

    ``|centroid|^2 + (trace(cov) - cov_trace_target)^2 + sum of squared
    off-diagonal covariance entries`` (all six entries, so each symmetric pair
    counts twice).  Axis order and mirror flips are not pinned.
    """
    _, centroid, cov = moments(fb, cfg)
    diagonal = torch.diagonal(cov, dim1=-2, dim2=-1)
    off_diagonal = cov.square().sum(dim=(-2, -1)) - diagonal.square().sum(dim=-1)
    trace_error = diagonal.sum(dim=-1) - _cov_trace_target(cfg)
    return (centroid.square().sum(dim=-1) + trace_error.square() + off_diagonal).mean()


POSE_KEYS: Tuple[str, ...] = ("pose_centroid", "pose_scale", "pose_axes")


def _half_moments(weight: Tensor, x: Tensor) -> Tuple[Tensor, Tensor]:
    """Centroid ``B x 3`` and covariance ``B x 3 x 3`` of one half of the samples.

    The covariance is taken about the half's own centroid and divided by
    ``1 - 1 / n_eff`` (``n_eff = (sum w)^2 / sum w^2``), the weighted form of
    the ``n - 1`` correction, so its expectation does not shrink with the
    number of inside samples.
    """
    total = weight.sum(dim=1) + _WEIGHT_EPS
    centroid = (weight.unsqueeze(-1) * x).sum(dim=1) / total.unsqueeze(-1)
    offset = x - centroid.unsqueeze(1)
    cov = torch.einsum("bni,bnj->bij", weight.unsqueeze(-1) * offset, offset) / total.reshape(-1, 1, 1)
    with torch.no_grad():
        effective = total.square() / (weight.square().sum(dim=1) + _WEIGHT_EPS)
        correction = 1.0 / (1.0 - 1.0 / effective.clamp_min(2.0))
    return centroid, cov * correction.reshape(-1, 1, 1)


def pose_terms(fb: FieldBatch, cfg) -> Dict[str, Tensor]:
    """Unbiased pose constraints from two disjoint halves of the volume samples.

    Squaring one noisy estimate adds its variance (``E|c_hat|^2 = |c|^2 +
    var``), which at ``N = 2048`` is ten times a useful tolerance.  Here every
    quantity is estimated twice, from alternate rows of the samples (each half
    a uniform sample of the box; row 0, the origin, is not a random sample and
    is left out), and the two independent estimates are multiplied:

    - ``pose_centroid``: ``c1 . c2``, expectation ``|centroid|^2``;
    - ``pose_scale``: ``(tr1 - T)(tr2 - T)``, expectation
      ``(trace(cov) - cov_trace_target)^2``;
    - ``pose_axes``: ``sum_{i != j} cov1_ij cov2_ij``, expectation the sum of
      squared off-diagonal covariance entries (each symmetric pair twice).

    Each is the mean over codes.  They fluctuate around the true value, so
    they can be slightly negative on a perfectly pinned shape.

    With ``cfg.pose_power`` p other than 2 the first two measure extent instead of mass
    (``extent_pose``): ``pose_centroid`` is the squared offset of the p-centre, and ``pose_scale`` is
    the squared error of ``size^2`` with ``size = (p-th moment)^(1/p)``: ``(s1 - T)(s2 - T)`` with
    ``s = M^(2/p)`` from each half (``_extent_size_sq``). Both keep the units, and so the tolerances,
    of the mass-based terms. ``pose_axes`` is the same for every p.

    The size error is NOT taken linear in the moment M. M goes like size^p, so it hardly changes
    when a small shape shrinks further: a term linear in M pulls a shape that is too small back with
    a force that vanishes like size^(p-2), and shapes then shrink to nothing (measured: half of the
    random codes decoded to an empty shape after 1,000 updates). ``M^(2/p)`` goes like size^2, as the
    covariance trace does, so the pull back grows with the error.
    """
    weight = soft_inside(fb, cfg)
    x = fb.x.detach()
    c1, cov1 = _half_moments(weight[:, 1::2], x[:, 1::2])
    c2, cov2 = _half_moments(weight[:, 2::2], x[:, 2::2])
    d1 = torch.diagonal(cov1, dim1=-2, dim2=-1)
    d2 = torch.diagonal(cov2, dim1=-2, dim2=-1)
    axes = (cov1 * cov2).sum(dim=(-2, -1)) - (d1 * d2).sum(dim=-1)
    power = float(getattr(cfg, "pose_power", 2.0))
    if power == 2.0:
        target = _cov_trace_target(cfg)
        centre = (c1 * c2).sum(dim=-1).mean()
        scale = ((d1.sum(dim=-1) - target) * (d2.sum(dim=-1) - target)).mean()
    else:
        o1, m1 = extent_pose(weight[:, 1::2], x[:, 1::2], power)
        o2, m2 = extent_pose(weight[:, 2::2], x[:, 2::2], power)
        moment_target = pose_moment_target(cfg)
        target = moment_target ** (2.0 / power)
        s1 = _extent_size_sq(weight[:, 1::2], x[:, 1::2], m1, power, moment_target)
        s2 = _extent_size_sq(weight[:, 2::2], x[:, 2::2], m2, power, moment_target)
        centre = (o1 * o2).sum(dim=-1).mean()
        scale = ((s1 - target) * (s2 - target)).mean()
    return {"pose_centroid": centre, "pose_scale": scale, "pose_axes": axes.mean()}


def _extent_size_sq(weight: Tensor, x: Tensor, moment: Tensor, power: float, moment_target: float) -> Tensor:
    """``size^2 = moment^(2/p)`` of each code from one half of the samples, ``B``.

    A root of a noisy mean reads low: ``E[M_hat^a] = M^a (1 - a (1 - a) v / 2)`` to first order, with
    ``v`` the relative variance of ``M_hat``. ``v`` is estimated from the same samples and divided out
    (it is not differentiated). A thin shape has only a few inside samples per half and still reads
    somewhat small, which errs towards growing it. Moments below that of a quarter-size shape are
    held at that value: an empty half has no samples to act on.
    """
    a = 2.0 / float(power)
    with torch.no_grad():
        rho_p = x.norm(dim=-1).pow(float(power))
        total = (weight * rho_p).sum(dim=1).clamp_min(1e-30)
        spread = (weight.square() * (rho_p - moment.unsqueeze(1)).square()).sum(dim=1) / total.square()
        low = (1.0 - 0.5 * a * (1.0 - a) * spread).clamp_min(0.5)
    floor = float(moment_target) * 0.25 ** float(power)
    return moment.clamp_min(floor).pow(a) / low


# ------------------------------------------------------------------- descriptors
def descriptors(fb: FieldBatch, cfg) -> Tensor:
    """Differentiable shape statistics, ``B x len(DESCRIPTOR_NAMES)``.

    All are computed from the volume samples already in ``fb``:

    - ``volume_fraction``: soft inside fraction of the box.
    - ``surface_area``: soft count of samples with ``|f| < t/2``, divided by
      the band width, times the box volume (exact for a unit-gradient field).
    - ``sphericity``: ``pi^(1/3) (6 V)^(2/3) / area``; 1 for a ball.
    - ``elongation``, ``flatness``: ``sqrt(l1 / l2)`` and ``sqrt(l2 / l3)`` of
      the covariance eigenvalues ``l1 >= l2 >= l3``; both 1 for a ball.
    - ``mean_depth``: mean of ``f`` over the inside, divided by the radius of
      the ball of equal volume; 1/4 for a ball.
    - ``offset_growth``: volume gained by ``{f > -t}`` divided by ``area * t``;
      1 for a flat wall, above 1 where convex, below 1 where concave.
    """
    f = fb.f
    tau = float(cfg.soft_tau)
    feature = float(cfg.min_feature)

    fraction, _, cov = moments(fb, cfg)
    volume = BOX_VOLUME * fraction

    half = 0.5 * feature
    soft = 0.25 * half
    # Integral over f of the soft band indicator: the effective band width.
    width = 2.0 * soft * math.log1p(math.exp(half / soft))
    in_band = torch.sigmoid((half - f.abs()) / soft)
    area = BOX_VOLUME * in_band.mean(dim=1) / width

    sphericity = math.pi ** (1.0 / 3.0) * (6.0 * volume + _POW_EPS) ** (2.0 / 3.0) / (area + _AREA_EPS)

    eig = torch.linalg.eigvalsh(cov).clamp_min(_EIG_FLOOR)  # ascending
    elongation = torch.sqrt(eig[:, 2] / eig[:, 1])
    flatness = torch.sqrt(eig[:, 1] / eig[:, 0])

    weight = torch.sigmoid(f / tau)
    depth = (weight * torch.relu(f)).sum(dim=1) / (weight.sum(dim=1) + _WEIGHT_EPS)
    radius = (3.0 * volume / (4.0 * math.pi) + _POW_EPS) ** (1.0 / 3.0)
    mean_depth = depth / radius

    grown = BOX_VOLUME * torch.sigmoid((f + feature) / tau).mean(dim=1)
    offset_growth = (grown - volume) / ((area + _AREA_EPS) * feature)

    # Thickness profile: the depth below which 25%, 50% and 90% of the shape's inside points
    # lie, relative to the equal-volume radius. A ball gives about 0.09, 0.21 and 0.54; thin
    # limbs pull the low levels down, a fat core pushes the high level up. Sorting is
    # differentiable with respect to the selected values.
    inside = f.detach() > 0
    count = inside.sum(dim=1)
    ranked = torch.sort(torch.where(inside, f, f.new_full((), -1.0)), dim=1, descending=True).values
    levels = []
    for q in _DEPTH_LEVELS:
        # the q-th percentile from the shallow end is at rank (1 - q) from the deep end
        pos = ((1.0 - q) * (count.to(f.dtype) - 1.0)).clamp_min(0.0)
        lo = pos.floor().long().clamp(max=f.shape[1] - 1)
        hi = (lo + 1).clamp(max=f.shape[1] - 1)
        hi = torch.minimum(hi, (count - 1).clamp_min(0))
        frac = (pos - lo.to(f.dtype)).unsqueeze(1)
        value = ranked.gather(1, lo.unsqueeze(1)) * (1.0 - frac) + ranked.gather(1, hi.unsqueeze(1)) * frac
        value = torch.where(count.unsqueeze(1) > 0, torch.relu(value), value * 0.0)
        levels.append(value.squeeze(1) / radius)

    return torch.stack(
        [fraction, area, sphericity, elongation, flatness, mean_depth, offset_growth, *levels], dim=-1
    )


# --------------------------------------------------------------------- diversity
def local_chamfer(fb: FieldBatch, cfg, generator: Optional[torch.Generator] = None
                  ) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Push the surfaces of codes in a cluster apart where they are locally far from each other.

    Uses only the surface points every step already has (the one-step guesses and the
    gradients at their source samples); no network evaluations.

    For an ordered pair of codes (A, B): for each of ``local_chamfer_queries`` surface points
    of A, the nearest of B's surface points is found, and the distance is measured to B's
    tangent plane there (to the plane, not the point, which removes the sideways error of one
    sample spacing). The same measurement from B's points to B's own other points gives the
    sampling-noise distribution; its ``local_chamfer_quantile`` level is the threshold. A point
    contributes ``leak * d + relu(d - threshold)``: differences spread thinly over the surface
    earn little, a feature that sticks out in one place earns a lot, and the small linear part
    keeps a gradient for pairs that are still nearly identical.

    The pair value is the mean over both directions; the loss is minus the logarithm of the
    latent-distance-weighted mean over pairs (the same pure push as the ``repel`` mode).
    Surface points move along their (detached) normals with the field value, so the gradient
    acts on the places where the two surfaces differ.
    """
    f = fb.f
    batch, count = f.shape
    device = f.device
    zero = f.sum() * 0.0
    nothing = {"local_distance": zero.detach(), "local_floor": zero.detach(), "local_above": zero.detach()}
    cluster = fb.cluster
    same = (cluster.unsqueeze(0) == cluster.unsqueeze(1)) & ~torch.eye(batch, dtype=torch.bool, device=device)
    qi, ri = torch.nonzero(same, as_tuple=True)                       # ordered pairs: query code, reference code
    if qi.numel() == 0:
        return zero, nothing

    g = fb.g.detach()
    length = torch.sqrt(g.square().sum(dim=-1, keepdim=True) + _NORM_EPS)
    normal = g / length                                               # unit, into the solid
    surface = fb.x.detach() - f.unsqueeze(-1) * normal / length       # one step along the normal
    with torch.no_grad():
        err = fb.fp.detach().abs()
        tol = 3.0 * err[:, 1:].median().clamp_min(1e-4)
        valid = (err <= tol) & (length.squeeze(-1) > 0.3)
        valid[:, 0] = False                                           # the origin row is not a surface sample
        queries = max(1, min(int(getattr(cfg, "local_chamfer_queries", 512)), count))
        score = torch.rand(batch, count, generator=generator,
                           device=generator.device if generator is not None else device).to(device)
        pick = (score + (~valid).to(score.dtype)).topk(queries, dim=1, largest=False).indices   # valid first
        pick_ok = valid.gather(1, pick)
        far = torch.where(valid, torch.zeros_like(err), torch.full_like(err, 1e6))              # B x N

    q_pts = surface.gather(1, pick.unsqueeze(-1).expand(-1, -1, 3))   # B x Q x 3, in the graph

    def nearest(query_code: Tensor, ref_code: Tensor, exclude_self: bool) -> Tuple[Tensor, Tensor]:
        """Index of the nearest valid reference point and the point-to-plane distance."""
        out_idx = []
        with torch.no_grad():
            for s in range(0, query_code.numel(), 32):
                qc, rc = query_code[s:s + 32], ref_code[s:s + 32]
                d = torch.cdist(q_pts[qc].detach(), surface[rc].detach()) + far[rc].unsqueeze(1)
                if exclude_self:
                    d = d + (d < 1e-7).to(d.dtype) * 1e6              # skip the point itself
                out_idx.append(d.argmin(dim=2))
        idx = torch.cat(out_idx)                                      # P x Q
        ref_pts = surface[ref_code].gather(1, idx.unsqueeze(-1).expand(-1, -1, 3))
        ref_n = normal[ref_code].gather(1, idx.unsqueeze(-1).expand(-1, -1, 3))
        return idx, ((q_pts[query_code] - ref_pts) * ref_n).sum(dim=-1).abs()

    with torch.no_grad():
        codes_all = torch.arange(batch, device=device)
        _, self_d = nearest(codes_all, codes_all, exclude_self=True)
        self_d = torch.where(pick_ok, self_d, torch.full_like(self_d, float("nan")))
        level = float(getattr(cfg, "local_chamfer_quantile", 0.75))
        floor = torch.nanquantile(self_d, level, dim=1).nan_to_num(0.0)   # B, noise floor of each surface

    _, d = nearest(qi, ri, exclude_self=False)                        # P x Q, in the graph
    weight = pick_ok[qi].to(f.dtype)
    leak = float(getattr(cfg, "local_chamfer_leak", 0.1))
    above = torch.relu(d - floor[ri].unsqueeze(1))
    per_pair = ((leak * d + above) * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)   # P

    codes = fb.codes.detach()
    latent = chord(codes[qi], codes[ri])
    pushed = (latent * per_pair).sum() / latent.square().sum().clamp_min(1e-12)
    loss = -torch.log(pushed / float(cfg.diversity_slope) + float(getattr(cfg, "repel_eps", 1e-3)))
    with torch.no_grad():
        info = {"local_distance": (d * weight).sum() / weight.sum().clamp_min(1.0),
                "local_floor": floor.mean(),
                "local_above": ((above > 0).to(f.dtype) * weight).sum() / weight.sum().clamp_min(1.0)}
    return loss, info


# Groups of shape statistics that are pushed apart separately (see statistic_repulsion). Only
# statistics with a usable signal-to-noise ratio at the training point count are included.
STAT_GROUPS: Dict[str, Tuple[str, ...]] = {
    "proportions": ("elongation", "flatness"),
    "thickness": ("mean_depth", "depth_p50", "depth_p90"),
    "compactness": ("sphericity", "offset_growth"),
}


def statistic_repulsion(fb: FieldBatch, cfg) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Push shapes apart in each group of statistics, more the farther apart their codes are.

    No target, no curve and no chosen latent direction: for each group the spread is the mean
    over all pairs of codes of ``chord * squared statistic difference`` (in units of each
    statistic's scale), and the loss is ``-log(spread)`` averaged over groups.

    - Weighting by latent distance leaves near codes alone, so locality is not disturbed (it is
      enforced by the field terms and the decoder's smoothness).
    - Squared differences: sampling noise in a statistic then adds a constant to the spread, so
      on average the gradient points towards real variation instead of towards noise.
    - The logarithm gives diminishing returns: a group that barely varies is pushed hard, one
      that varies a lot only weakly, and a noise-dominated group less. No per-group weights.
    """
    stats = descriptors(fb, cfg)
    batch = stats.shape[0]
    zero = stats.sum() * 0.0
    info = {f"spread_{name}": zero.detach() for name in STAT_GROUPS}
    if batch < 2:
        return zero, info
    scales = torch.as_tensor(DESCRIPTOR_SCALES, dtype=stats.dtype, device=stats.device)
    codes = fb.codes.detach()
    weight = torch.triu(chord(codes.unsqueeze(1), codes.unsqueeze(0)), diagonal=1)
    weight = weight / weight.sum().clamp_min(1e-12)
    scaled = stats / scales
    eps = float(getattr(cfg, "stat_repulsion_eps", 1e-3))
    total = zero
    for name, members in STAT_GROUPS.items():
        index = [DESCRIPTOR_NAMES.index(m) for m in members]
        part = scaled[:, index]
        gap = (part.unsqueeze(1) - part.unsqueeze(0)).square().mean(dim=-1)      # B x B
        group_spread = (weight * gap).sum()
        total = total - torch.log(group_spread + eps)
        info[f"spread_{name}"] = group_spread.detach()
    return total / len(STAT_GROUPS), info


def spread(fb: FieldBatch, cfg) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Different latent directions must change the shape in different ways.

    The pair loss in :func:`diversity` only checks how *much* two codes differ. It is
    satisfied in part by a family in which every latent direction produces the same change
    in different amounts (measured: 99.9% of the surface motion along one direction).

    Here each cluster's first code is its centre. For every other member ``k`` the pattern
    ``P_k = (f_k - f_centre) / chord_k`` is the field change per unit latent distance (pose
    removed when ``diversity_pose_invariant``). For two members of one cluster, the
    correlation of their patterns over the shared sample points should equal the cosine of
    the angle between their latent offsets: steps at right angles in latent space should
    give unrelated changes, the same step the same change. That is what an embedding which
    preserves distances does, written with inner products instead of lengths.

    The loss is the mean squared gap between pattern correlation and latent cosine. Pattern
    norms get a floor (``spread_floor`` times the target slope), so a direction that barely
    moves the shape reads as uncorrelated instead of amplifying noise; reviving such a
    direction is the job of the pair loss.
    """
    f = fb.f
    batch = f.shape[0]
    device = f.device
    cluster = fb.cluster
    order = torch.arange(batch, device=device)
    num_clusters = int(cluster.max()) + 1 if batch > 0 else 0
    first = torch.full((num_clusters,), batch, dtype=torch.long, device=device)
    first = first.scatter_reduce(0, cluster, order, reduce="amin")
    centre = first[cluster]
    member = (order != centre).nonzero().squeeze(1)
    zero = f.sum() * 0.0
    nothing = {"pairs": zero.detach(), "corr_abs_mean": zero.detach(), "cos_abs_mean": zero.detach()}
    if member.numel() < 2:
        return zero, nothing
    c = centre[member]
    x = fb.x.detach()
    mask = (x[member].square().sum(dim=-1) < float(cfg.bound_radius) ** 2).to(f.dtype)
    delta = f[member] - f[c]
    if bool(getattr(cfg, "diversity_pose_invariant", False)):
        delta = _remove_pose(delta, x[member], 0.5 * (fb.g[member] + fb.g[c]).detach(),
                             0.5 * (f[member] + f[c]).detach(), mask)
    codes = fb.codes.detach()
    offset = codes[member] - codes[c]
    length = offset.norm(dim=-1).clamp_min(1e-6)
    pattern = delta * mask / length.unsqueeze(-1)                       # M x N, per unit chord
    count = mask.sum(dim=1).clamp_min(1.0)

    group = cluster[member]
    same = group.unsqueeze(0) == group.unsqueeze(1)
    upper = torch.triu(torch.ones_like(same), diagonal=1)
    a, b = torch.nonzero(same & upper, as_tuple=True)
    if a.numel() == 0:
        return zero, nothing
    inner = (pattern[a] * pattern[b]).sum(dim=1) / count[a]
    power = pattern.square().sum(dim=1) / count
    floor = (float(getattr(cfg, "spread_floor", 0.1)) * float(cfg.diversity_slope)) ** 2
    corr = inner / torch.sqrt((power[a] + floor) * (power[b] + floor))
    cosine = (offset[a] * offset[b]).sum(dim=-1) / (length[a] * length[b])
    loss = (corr - cosine).square().mean()
    with torch.no_grad():
        info = {"pairs": torch.as_tensor(float(a.numel()), device=device),
                "corr_abs_mean": corr.abs().mean(), "cos_abs_mean": cosine.abs().mean()}
    return loss, info


def nearest_pool(fb: FieldBatch, pool_f: Tensor, index: Tensor, cfg,
                 pool_codes: Optional[Tensor] = None, pool_terms: bool = False,
                 norm: Optional[float] = None, slope_angles: Optional[Tensor] = None) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Nearest-neighbour repulsion against a large pool of codes, at the memory cost of the batch.

    ``fb`` holds the ``B`` codes that carry gradients; all share their sample points.
    ``pool_f`` (``P x M``) are the fields of ``P`` further random codes at the ``M`` sample
    points ``index`` (a subset of the batch's points), evaluated without gradients. Each of the
    ``B`` shapes is pushed away from its ``nearest_k`` most similar shapes among the other
    ``B - 1`` and the pool: candidates are ranked by the plain field distance, then the chosen
    pairs get the pose-corrected distance (using the batch code's own gradients for the fit) and
    a soft minimum. A pool neighbour is a fixed target for this step; a batch neighbour moves too.

    "Nearest of 32" is a coarse sample of a 31-dimensional family; the pool makes it "nearest
    of ``B + P - 1``" without holding the pool's gradients in memory.

    ``pool_codes`` (``P x D``, optional) are the pool's codes, in the order of ``pool_f``. They
    do not affect the loss; they only let the info report latent distances.

    The info is computed without gradients, over the chosen ``B x k`` neighbour pairs only (not
    over all pairs of the batch, as :func:`diversity` reports), so the trainer can log it
    without a separate in-batch pass:

    * ``nearest_distance``: mean over batch codes of the smallest pose-corrected distance among
      its chosen neighbours;
    * ``raw_nearest_distance``: the same with the plain (uncorrected) field distance, over all candidates;
    * ``pool_neighbour_share``: fraction of chosen neighbours that come from the pool;
    * ``mean_shape_distance``: mean pose-corrected distance over the chosen pairs;
    * ``mean_latent_distance``: mean chord distance between each batch code and its chosen
      neighbours (0.0 when ``pool_codes`` is not given);
    * ``mean_ratio``: mean of ``distance / (diversity_slope * chord)`` over the chosen pairs
      (0.0 when ``pool_codes`` is not given);
    * ``dead_zone_fraction``: fraction of chosen pairs with that ratio >= 1, i.e. above the old
      floor level, as the other no-floor modes report (0.0 when ``pool_codes`` is not given);
    * ``phase2`` and ``order_violations``: always 0.0 -- not measured in pooled mode.
    """
    # f_move (when set) has f's values; its gradient also moves the surface each point belongs to
    f = (fb.f if getattr(fb, "f_move", None) is None else fb.f_move)[:, index]   # B x M, in the graph
    g = fb.g[:, index].detach()
    x = fb.x[:, index].detach()
    batch, points = f.shape
    mask = (x.square().sum(dim=-1) < float(cfg.bound_radius) ** 2).to(f.dtype)
    count = mask.sum(dim=1).clamp_min(1.0)
    candidates = torch.cat([f, pool_f.to(f.dtype)], dim=0)            # (B + P) x M; batch rows keep gradients
    k = max(1, min(int(getattr(cfg, "nearest_k", 8)), candidates.shape[0] - 1))
    band = float(getattr(cfg, "variety_surface_band", 0.0))

    def near(a: Tensor, b: Tensor, inside: Tensor) -> Tensor:
        """Weights of a pair's points: all points inside the bound radius, or with a surface band
        only those within the band of either shape's surface. There the field difference is the
        distance from one surface to the other, so the pair's distance becomes a surface distance,
        which does not depend on how much volume a feature has. A pair with fewer than 8 such
        points keeps all its points."""
        inside = inside.expand(torch.broadcast_shapes(a.shape, b.shape))
        if band <= 0.0:
            return inside
        close = inside * (torch.minimum(a.abs(), b.abs()) < band).to(inside.dtype)
        enough = close.sum(dim=-1, keepdim=True) >= 8.0
        return torch.where(enough, close, inside)

    with torch.no_grad():
        raw = torch.empty((batch, candidates.shape[0]), dtype=f.dtype, device=f.device)
        for s in range(0, candidates.shape[0], 64):                   # chunked: B x 64 x M at a time
            mine, theirs = f.detach().unsqueeze(1), candidates[s:s + 64].detach().unsqueeze(0)
            weight = near(mine, theirs, mask.unsqueeze(1))
            raw[:, s:s + 64] = ((mine - theirs).abs() * weight).sum(dim=-1) / weight.sum(dim=-1).clamp_min(1.0)
        raw[torch.arange(batch, device=f.device), torch.arange(batch, device=f.device)] = float("inf")
        chosen = raw.topk(k, dim=1, largest=False).indices            # B x k
    other = candidates[chosen]                                        # B x k x M
    delta = (f.unsqueeze(1) - other).reshape(batch * k, points)
    pair_mask = near(f.detach().unsqueeze(1), other.detach(), mask.unsqueeze(1)).reshape(batch * k, points)
    if bool(getattr(cfg, "diversity_pose_invariant", False)):
        delta = _remove_pose(delta,
                             x.unsqueeze(1).expand(-1, k, -1, -1).reshape(batch * k, points, 3),
                             g.unsqueeze(1).expand(-1, k, -1, -1).reshape(batch * k, points, 3),
                             (0.5 * (f.unsqueeze(1) + other)).detach().reshape(batch * k, points), pair_mask)
    dist = ((delta.abs() * pair_mask).sum(dim=1) / pair_mask.sum(dim=1).clamp_min(1.0)).reshape(batch, k)
    if slope_angles is not None:
        dist = dist + cfg.slope_diversity_weight * slope_angles[:batch].gather(1, chosen)
    power = float(getattr(cfg, "nearest_power", 4.0))
    soft = (dist + _TARGET_FLOOR).pow(-power).sum(dim=1).pow(-1.0 / power)
    slope, eps = float(cfg.diversity_slope), float(getattr(cfg, "repel_eps", 1e-3))
    terms = -torch.log(soft / slope + eps)
    loss = terms.mean()
    pool = pool_f.to(f.dtype)
    if pool_terms and pool.shape[0] > 0:
        # The pool shapes' own terms, as functions of the batch fields. A batch shape is a
        # neighbour of pool shapes too, and that dependence is the other half of its gradient.
        # With these terms, one call returns the complete derivative of the whole-set loss with
        # respect to the batch fields; summed over the passes of an update (each pass taking its
        # turn as "batch", the loss divided by the number of passes) that is the exact gradient
        # of the nearest-neighbour loss over all the update's shapes. Scaled as sum / B so that
        # this division by the number of passes gives the mean over the whole set. When the
        # passes hold different numbers of shapes, ``norm`` (the mean number per pass) replaces
        # B, or a pass with few shapes would count each of them too heavily.
        size = pool.shape[0]
        row_mask, row_count = mask[:1], count[:1]                     # sample points are shared by all
        with torch.no_grad():
            raw_p = torch.empty((size, candidates.shape[0]), dtype=f.dtype, device=f.device)
            for s in range(0, candidates.shape[0], 64):
                mine, theirs = pool.unsqueeze(1), candidates[s:s + 64].detach().unsqueeze(0)
                weight = near(mine, theirs, row_mask.unsqueeze(1))
                raw_p[:, s:s + 64] = ((mine - theirs).abs() * weight).sum(dim=-1) / weight.sum(dim=-1).clamp_min(1.0)
            own = torch.arange(size, device=f.device)
            raw_p[own, batch + own] = float("inf")
            chosen_p = raw_p.topk(k, dim=1, largest=False).indices    # P x k
        other_p = candidates[chosen_p]                                # P x k x M; batch rows keep gradients
        delta_p = (pool.unsqueeze(1) - other_p).reshape(size * k, points)
        mask_p = near(pool.unsqueeze(1), other_p.detach(), row_mask.unsqueeze(1)).reshape(size * k, points)
        if bool(getattr(cfg, "diversity_pose_invariant", False)):
            # the fit uses the batch neighbour's gradients; a pool-pool pair has none and is left as is
            g_all = torch.cat([g, g.new_zeros((size, points, 3))], dim=0)
            delta_p = _remove_pose(delta_p, x[:1].expand(size * k, -1, -1), g_all[chosen_p].reshape(size * k, points, 3),
                                   (0.5 * (pool.unsqueeze(1) + other_p)).detach().reshape(size * k, points), mask_p)
        dist_p = ((delta_p.abs() * mask_p).sum(dim=1) / mask_p.sum(dim=1).clamp_min(1.0)).reshape(size, k)
        if slope_angles is not None:
            # The owner pass replays the angular derivative for both endpoints.
            dist_p = dist_p + cfg.slope_diversity_weight * slope_angles[batch:].detach().gather(1, chosen_p)
        soft_p = (dist_p + _TARGET_FLOOR).pow(-power).sum(dim=1).pow(-1.0 / power)
        loss = (terms.sum() - torch.log(soft_p / slope + eps).sum()) / (batch if norm is None else float(norm))
    with torch.no_grad():
        info = {"nearest_distance": dist.min(dim=1).values.mean(),
                "pool_neighbour_share": (chosen >= batch).to(f.dtype).mean(),
                "raw_nearest_distance": raw.min(dim=1).values.mean(),
                "mean_shape_distance": dist.mean(),
                "mean_latent_distance": dist.new_zeros(()),
                "mean_ratio": dist.new_zeros(()),
                "dead_zone_fraction": dist.new_zeros(()),
                "phase2": dist.new_zeros(()),
                "order_violations": dist.new_zeros(())}
        if pool_codes is not None:
            codes = fb.codes.detach()
            every = torch.cat([codes, pool_codes.detach().to(codes)], dim=0)      # same order as ``candidates``
            latent = chord(codes.unsqueeze(1), every[chosen]).to(dist.dtype)       # B x k
            ratio = dist / (float(cfg.diversity_slope) * latent).clamp_min(1e-6)
            info["mean_latent_distance"] = latent.mean()
            info["mean_ratio"] = ratio.mean()
            info["dead_zone_fraction"] = (ratio >= 1.0).to(dist.dtype).mean()
    return loss, info


def pair_push(fb: FieldBatch, first: Tensor, second: Tensor, chord_length, cfg
              ) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Push apart chosen pairs of codes that sit a known, small chord apart.

    ``first`` and ``second`` index rows of ``fb`` that share their sample points; ``chord_length``
    is the chord between the two codes of each pair (a float or one value per pair). The pair's
    distance is the mean absolute field difference inside the bound radius, with translation,
    rotation and resize removed when ``cfg.diversity_pose_invariant`` (fitted with the first
    code's gradients), and its rate is ``distance / (diversity_slope * chord)``. Each pair
    contributes ``-log(rate + weak_pair_eps)``: the push on a pair is about one over its current
    rate, so it is strongest on pairs whose codes differ along a direction the decoder ignores,
    and ``weak_pair_eps`` caps it. Returns the SUM over pairs and, without gradients, the mean
    and smallest rate.

    Used for pairs that differ only along the least-used directions of code space: nothing else
    separates such a pair, so the whole push lands on that direction.
    """
    f = fb.f[:, 1:]                                                    # row 0 is the origin
    x = fb.x[:, 1:].detach()
    g = fb.g[:, 1:].detach()
    fa, fo = f[first], f[second]
    xa = x[first]
    mask = (xa.square().sum(dim=-1) < float(cfg.bound_radius) ** 2).to(f.dtype)
    delta = fa - fo
    if bool(getattr(cfg, "diversity_pose_invariant", False)):
        delta = _remove_pose(delta, xa, g[first], (0.5 * (fa + fo)).detach(), mask)
    dist = (delta.abs() * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    length = torch.as_tensor(chord_length, dtype=f.dtype, device=f.device)
    rate = dist / (float(cfg.diversity_slope) * length).clamp_min(1e-9)
    terms = -torch.log(rate + float(getattr(cfg, "weak_pair_eps", 0.05)))
    with torch.no_grad():
        info = {"pair_rate": rate.mean(), "pair_rate_min": rate.min()}
    return terms.sum(), info


def frame_volume(fb: FieldBatch, centre: int, partners: Tensor, chord_length, used: int, cfg
                 ) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Log-volume of the field changes of a frame of partner codes around one code.

    ``partners`` are rows of ``fb`` that share their sample points with row ``centre``; each is
    ``chord_length`` away from it (one value per partner, or a float). The first ``used``
    partners lie along directions the decoder already uses, the others along unused ones. Each
    partner's field change per unit chord (pose removed with the centre's gradients when
    ``cfg.diversity_pose_invariant``) is one row; ``G`` is the Gram matrix of the rows (mean
    over the sample points inside the bound radius) and the term is
    ``-log det(G + weak_frame_sigma^2 I) / partners``.

    The determinant is the squared volume the changes span. A partner whose change is a
    multiple or mix of the others' adds no volume however large it is, so making an unused
    direction replay an existing pattern earns nothing; only a change that the other partners
    cannot produce does. The push on such a change is about one over its own size (plus the
    sigma floor), so the weakest new patterns are pushed hardest.

    The info (no gradients): ``new_rms`` is the RMS size of the unused-direction partners'
    changes after removing what the used-direction partners explain (the part that counts),
    ``new_raw_rms`` the same before removal, ``used_rms_min`` the smallest used-direction change.
    """
    f = fb.f[:, 1:]                                                    # row 0 is the origin
    x = fb.x[centre, 1:].detach()
    g = fb.g[centre, 1:].detach()
    count = int(partners.numel())
    mask = (x.square().sum(dim=-1) < float(cfg.bound_radius) ** 2).to(f.dtype)
    length = torch.as_tensor(chord_length, dtype=f.dtype, device=f.device).expand(count).clamp_min(1e-9)
    moved, base = f[partners], f[centre].unsqueeze(0)
    delta = (moved - base) / length.unsqueeze(-1)                      # P x N
    if bool(getattr(cfg, "diversity_pose_invariant", False)):
        delta = _remove_pose(delta, x.expand(count, -1, -1), g.expand(count, -1, -1),
                             (0.5 * (moved + base)).detach(), mask.expand(count, -1))
    delta = delta * mask
    gram = (delta @ delta.T / mask.sum().clamp_min(1.0)).double()
    eye = torch.eye(count, dtype=gram.dtype, device=gram.device)
    floor = float(getattr(cfg, "weak_frame_sigma", 0.02)) ** 2
    loss = (-torch.logdet(gram + floor * eye) / count).to(f.dtype)
    with torch.no_grad():
        zero = f.new_zeros(())
        info = {"new_rms": zero, "new_raw_rms": zero, "used_rms_min": zero}
        size = gram.diagonal().clamp_min(0.0).sqrt()
        # the frame's own spectrum: independent sizes of change, largest first
        spectrum = torch.linalg.eigvalsh(gram).clamp_min(0.0).flip(0)
        part = spectrum / spectrum.sum().clamp_min(1e-30)
        info["frame_effective"] = torch.exp(-(part * torch.log(part + 1e-30)).sum()).to(f.dtype)
        info["frame_above_floor"] = (spectrum > floor).sum().to(f.dtype)
        info["frame_rms_median"] = spectrum.median().sqrt().to(f.dtype)
        info["frame_rms_top"] = spectrum[0].sqrt().to(f.dtype)
        if used > 0:
            info["used_rms_min"] = size[:used].min().to(f.dtype)
        if count > used:
            rest = gram[used:, used:]
            if used > 0:
                cross = gram[used:, :used]
                rest = rest - cross @ torch.linalg.solve(gram[:used, :used] + floor * eye[:used, :used], cross.T)
            info["new_rms"] = rest.diagonal().clamp_min(0.0).sqrt().mean().to(f.dtype)
            info["new_raw_rms"] = size[used:].mean().to(f.dtype)
    return loss, info


def sample_grid(grids: Tensor, points: Tensor) -> Tensor:
    """Trilinear values of ``A`` fields stored on grids, at ``A x N x 3`` points.

    ``grids`` is ``A x R x R x R`` with ``grids[a, i, j, l]`` the value at ``(x_i, y_j, z_l)`` on
    ``linspace(-1, 1, R)``. Points outside the box read the nearest boundary value. The result
    is differentiable with respect to ``points``.
    """
    # grid_sample indexes the last grid dimension with the first coordinate, hence the flip
    query = points.flip(-1)[:, :, None, None, :]
    values = torch.nn.functional.grid_sample(grids.unsqueeze(1), query, mode="bilinear",
                                             padding_mode="border", align_corners=True)
    return values.reshape(grids.shape[0], points.shape[1])


def anchor_match(fb: FieldBatch, rows: Tensor, grids: Tensor, cfg, weights: Optional[Tensor] = None
                 ) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Pull the decoder's field at some codes onto given target fields.

    ``weights`` (one per row, optional) scales each row's pull, for rows that sit some way from
    their anchor's code; the reported parts are then weighted means, so they describe the rows
    nearest their anchors most.

    ``rows`` are rows of ``fb`` and ``grids`` (``A x R x R x R``, positive inside, see
    :func:`sample_grid`) their targets. Two parts, both read at points the batch already has:

    * value: the mean absolute difference between the field and the target at the sample points
      inside the bound radius. This is the same distance the diversity terms use between two
      shapes, here made small instead of large.
    * surface: the target's value at the code's own surface guesses (``fb.xp``) that started
      within ``anchor_surface_band`` of the surface; zero when the decoder's surface lies on the
      target's. Weighted by ``anchor_surface_weight``.

    Returns the mean over the rows and, without gradients, the two parts and the overlap of the
    two solids (intersection over union at the sample points).
    """
    f = (fb.f if getattr(fb, "f_move", None) is None else fb.f_move)[rows]
    x = fb.x[rows].detach()
    mask = (x.square().sum(dim=-1) < float(cfg.bound_radius) ** 2).to(f.dtype)
    grids = grids.to(f.dtype)
    with torch.no_grad():
        target = sample_grid(grids, x)
    value = ((f - target).abs() * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    near = mask * (f.detach().abs() < float(getattr(cfg, "anchor_surface_band", 0.1))).to(f.dtype)
    surface = (sample_grid(grids, fb.xp[rows]).abs() * near).sum(dim=1) / near.sum(dim=1).clamp_min(1.0)
    scale = torch.ones_like(value) if weights is None else weights.to(value)
    loss = (scale * (value + float(getattr(cfg, "anchor_surface_weight", 1.0)) * surface)).mean()
    with torch.no_grad():
        ours, theirs = f > 0, target > 0
        union = (ours | theirs).sum(dim=1).clamp_min(1)
        share = scale / scale.sum().clamp_min(1e-9)
        info = {"distance": (share * value).sum(), "surface": (share * surface).sum(),
                "iou": (share * ((ours & theirs).sum(dim=1) / union).to(f.dtype)).sum()}
    return loss, info


def _order_violations(distance: Tensor, latent: Tensor, first: Tensor, second: Tensor,
                      batch: int, tie: float) -> Tensor:
    """Fraction of (anchor, nearer, farther) triples whose shape distances are the wrong way round."""
    dist_m = distance.new_zeros((batch, batch))
    lat_m = distance.new_zeros((batch, batch))
    ok = torch.zeros((batch, batch), dtype=torch.bool, device=distance.device)
    dist_m[first, second] = distance
    dist_m[second, first] = distance
    lat_m[first, second] = latent
    lat_m[second, first] = latent
    ok[first, second] = True
    ok[second, first] = True
    triple = ok.unsqueeze(2) & ok.unsqueeze(1) & (lat_m.unsqueeze(2) < (1.0 - tie) * lat_m.unsqueeze(1))
    if not bool(triple.any()):
        return distance.new_zeros(())
    return (dist_m.unsqueeze(2) > dist_m.unsqueeze(1))[triple].to(distance.dtype).mean()


def _remove_pose(delta: Tensor, x: Tensor, g: Tensor, value: Tensor, mask: Tensor,
                 weights: Optional[Tensor] = None) -> Tensor:
    """Remove from a field difference the part a small rigid motion or resize explains.

    Moving a shape by a small translation ``a``, rotation ``w`` and uniform scale ``1 + s``
    changes its distance field at ``x`` by about ``-g.a - w.(x cross g) + s (f - g.x)``, with
    ``g`` the spatial gradient (a resized shape's distances are resized too, hence the ``f``).
    Those seven fields (three translations, three rotations, one scale)
    are fitted to ``delta`` by masked, lightly ridged least squares over the shared sample
    points, and the fit is subtracted. The basis is detached, so the result is a fixed
    linear projection of ``delta``: its gradient has no component along any pose motion,
    and translating, turning or resizing a shape earns no shape distance.

    ``delta``, ``value`` (the pair's mean field) and ``mask`` are ``P x N``; ``x`` and ``g`` are ``P x N x 3``.
    Exact to first order only, which suits the small clusters the pair loss uses.
    """
    resize = value.unsqueeze(-1) - (g * x).sum(dim=-1, keepdim=True)
    basis = torch.cat([g, torch.linalg.cross(x, g, dim=-1), resize], dim=-1)
    basis = basis * mask.unsqueeze(-1)                                   # P x N x 7
    weighted = basis if weights is None else basis * weights.unsqueeze(-1)
    gram = weighted.transpose(1, 2) @ basis                              # P x 7 x 7
    scale = torch.diagonal(gram, dim1=-2, dim2=-1).mean(dim=-1).clamp_min(1e-12)
    eye = torch.eye(7, dtype=gram.dtype, device=gram.device)
    gram = gram + (1e-4 * scale)[:, None, None] * eye
    rhs = weighted.transpose(1, 2) @ (delta * mask).unsqueeze(-1)        # P x 7 x 1
    coeff = torch.linalg.solve(gram, rhs)
    return delta - (basis @ coeff).squeeze(-1)


def diversity(fb: FieldBatch, cfg, step: Optional[int] = None) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Match shape distance to latent distance inside each cluster.

    For every pair of codes of one cluster, the shape distance is the mean
    absolute field difference over the shared samples with ``|x| < R_b`` and
    the target is ``diversity_slope * chord``.  The loss per pair is
    ``relu(|d / target - 1| - diversity_tolerance) ** diversity_power``: zero
    inside the dead zone, steep outside it.  From ``phase2_start_step`` on the
    shape distance also includes scaled descriptor differences.

    ``info`` holds detached scalars: ``mean_ratio``, ``dead_zone_fraction``,
    ``pairs``, ``mean_shape_distance``, ``mean_latent_distance`` and
    ``phase2``.  With no pair in the batch the loss is a zero that is still
    connected to ``fb.f`` and the info values are zero.
    """
    f = fb.f
    batch = f.shape[0]
    cluster = fb.cluster
    same = cluster.unsqueeze(0) == cluster.unsqueeze(1)
    upper = torch.triu(torch.ones(batch, batch, dtype=torch.bool, device=f.device), diagonal=1)
    first, second = torch.nonzero(same & upper, as_tuple=True)
    pairs = first.numel()
    phase2 = step is not None and step >= int(cfg.phase2_start_step)
    if pairs == 0:
        zero = f.sum() * 0.0
        nothing = zero.detach()
        info = {
            "mean_ratio": nothing,
            "dead_zone_fraction": nothing,
            "pairs": nothing,
            "order_violations": nothing,
            "nearest_distance": nothing,
            "mean_shape_distance": nothing,
            "mean_raw_distance": nothing,
            "pose_fraction": nothing,
            "mean_latent_distance": nothing,
            "phase2": torch.as_tensor(float(phase2), device=f.device),
        }
        return zero, info

    bounded = (fb.x.detach().square().sum(dim=-1) < float(cfg.bound_radius) ** 2).to(f.dtype)
    mask = bounded[first]
    delta = f[first] - f[second]
    raw_distance = (delta.abs() * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    if bool(getattr(cfg, "diversity_pose_invariant", False)):
        delta = _remove_pose(delta, fb.x.detach()[first], 0.5 * (fb.g[first] + fb.g[second]).detach(),
                             0.5 * (f[first] + f[second]).detach(), mask)
    difference = delta.abs()
    distance = (difference * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    if phase2:
        stats = descriptors(fb, cfg)
        scales = torch.as_tensor(DESCRIPTOR_SCALES, dtype=stats.dtype, device=stats.device)
        gap = ((stats[first] - stats[second]).abs() / scales).mean(dim=1)
        distance = distance + float(cfg.descriptor_weight) * gap

    codes = fb.codes.detach()
    latent = chord(codes[first], codes[second])
    target = (float(cfg.diversity_slope) * latent).clamp_min(_TARGET_FLOOR)
    ratio = distance / target
    order_violations = f.new_zeros(())
    mode = str(getattr(cfg, "diversity_mode", "slope"))
    nearest_distance = f.new_zeros(())
    if mode == "nearest":
        # Every shape is pushed away from the shapes most like it in the batch. Unlike a push on
        # the total, this is dominated by near neighbours, so it fills the available variety
        # evenly instead of piling onto the largest existing difference, and it rewards extra
        # independent ways of varying (a family with few dials is crowded: every shape has a
        # near twin). No floor, no ordering, no slope; meant for globally random codes.
        dist_m = f.new_full((batch, batch), float("inf"))
        dist_m[first, second] = distance
        dist_m[second, first] = distance
        has = torch.isfinite(dist_m).any(dim=1)
        power = float(getattr(cfg, "nearest_power", 4.0))
        # soft minimum over the other shapes (a power mean): the nearest dominates, the next few count
        soft = (dist_m[has] + _TARGET_FLOOR).pow(-power).sum(dim=1).pow(-1.0 / power)
        excess = torch.relu(1.0 - ratio)                 # only for the "above floor" statistic
        loss = -torch.log(soft / float(cfg.diversity_slope) + float(getattr(cfg, "repel_eps", 1e-3))).mean()
        nearest_distance = dist_m[has].min(dim=1).values.mean().detach()
    if mode in ("ordinal", "repel", "nearest"):
        # reported for every mode that does not enforce a curve: how often a code that is
        # nearer in latent space is farther in shape
        with torch.no_grad():
            order_violations = _order_violations(distance, latent, first, second, batch,
                                                 float(getattr(cfg, "ordinal_tie", 0.1)))
    if mode == "repel":
        # Pure push, no floor, no ordering, no curve: maximise the latent-distance-weighted mean
        # shape distance, with a logarithm for diminishing returns. Weighting by latent distance
        # leaves near codes alone (their similarity comes from the decoder's smoothness) and
        # already favours pairing larger latent distances with larger shape distances.
        excess = torch.relu(1.0 - ratio)                 # only for the "above floor" statistic
        pushed = (latent * distance).sum() / latent.square().sum().clamp_min(1e-12)
        loss = -torch.log(pushed / float(cfg.diversity_slope) + float(getattr(cfg, "repel_eps", 1e-3)))
    elif mode == "ordinal":
        # Floor only: at least ``diversity_slope`` of shape distance per unit latent distance,
        # no upper limit and no curve to follow.
        excess = torch.relu(1.0 - ratio)
        loss = excess.pow(float(cfg.diversity_power)).mean()
        # Ordering: seen from any code, a code that is nearer in latent space must be nearer in
        # shape. Compared as a relative gap, so it does not depend on the overall size; pairs of
        # near-equal latent distance are skipped. Uses only the pair distances computed above.
        dist_m = f.new_zeros((batch, batch))
        lat_m = f.new_zeros((batch, batch))
        ok = torch.zeros((batch, batch), dtype=torch.bool, device=f.device)
        dist_m[first, second] = distance
        dist_m[second, first] = distance
        lat_m[first, second] = latent
        lat_m[second, first] = latent
        ok[first, second] = True
        ok[second, first] = True
        near_d, far_d = dist_m.unsqueeze(2), dist_m.unsqueeze(1)          # [anchor, near, far]
        near_c, far_c = lat_m.unsqueeze(2), lat_m.unsqueeze(1)
        tie = float(getattr(cfg, "ordinal_tie", 0.1))
        triple = ok.unsqueeze(2) & ok.unsqueeze(1) & (near_c < (1.0 - tie) * far_c)
        if bool(triple.any()):
            gap_c = (far_c - near_c) / (far_c + near_c).clamp_min(1e-12)
            gap_d = (far_d - near_d) / (far_d + near_d + _TARGET_FLOOR)
            short = torch.relu(float(getattr(cfg, "ordinal_margin", 0.25)) * gap_c - gap_d)
            loss = loss + float(getattr(cfg, "ordinal_weight", 1.0)) * short[triple].square().mean()
    elif mode != "nearest":
        excess = torch.relu((ratio - 1.0).abs() - float(cfg.diversity_tolerance))
        loss = excess.pow(float(cfg.diversity_power)).mean()

    with torch.no_grad():
        info = {
            "order_violations": order_violations,
            "nearest_distance": nearest_distance,
            "mean_ratio": ratio.mean(),
            "dead_zone_fraction": (excess == 0).to(f.dtype).mean(),
            "pairs": torch.as_tensor(float(pairs), device=f.device),
            "mean_shape_distance": distance.mean(),
            "mean_raw_distance": raw_distance.mean(),
            "pose_fraction": 1.0 - (distance.detach().mean() / raw_distance.mean().clamp_min(1e-12)),
            "mean_latent_distance": latent.mean(),
            "phase2": torch.as_tensor(float(phase2), device=f.device),
        }
    return loss, info
