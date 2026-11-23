"""Field evaluation shared by every loss: values, spatial gradients, surface guesses."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
from torch import Tensor

from .config import DataFreeConfig
from .latent import _device_of, _rand, _randn

SHELL_LIMITS = (0.0, 0.2, 0.55, 0.7)    # inner edges of the four bands; the last ends at bound_radius
SHELL_SHARES = (0.05, 0.50, 0.20)        # the outer band takes the rest


def _shell_counts(n: int):
    counts = [round(share * n) for share in SHELL_SHARES]
    counts.append(n - sum(counts))
    return counts


def shell_volume_weight(x: Tensor, cfg: DataFreeConfig) -> Tensor:
    """Per-sample weights (shape of ``x`` without the last axis) for shell samples: the ratio of the
    box's uniform density to the shell density at that radius, ``V_band / (8 * share_band)``. A mean
    of ``weight * h`` over shell samples estimates the mean of ``h`` over the box ``[-1, 1]^3`` (for
    ``h`` zero beyond ``bound_radius``). Points beyond ``bound_radius`` get weight 0."""
    n = int(cfg.points)
    counts = _shell_counts(n)
    edges = (*SHELL_LIMITS, float(cfg.bound_radius))
    r = x.detach().norm(dim=-1)
    weight = torch.zeros_like(r)
    for count, low, high in zip(counts, edges, edges[1:]):
        band = (r >= low) & (r < high) if high < edges[-1] else (r >= low) & (r <= high + 1e-6)
        value = (4.0 / 3.0) * math.pi * (high ** 3 - low ** 3) / (8.0 * count / n)
        weight = torch.where(band, torch.full_like(r, value), weight)
    return weight


def shell_points(cfg: DataFreeConfig, clusters: int, generator, device) -> Tensor:
    """``C×N×3`` samples in four radial bands out to ``bound_radius`` (see ``point_sampling``)."""
    n = int(cfg.points)
    counts = _shell_counts(n)
    edges = (*SHELL_LIMITS, float(cfg.bound_radius))
    direction = torch.nn.functional.normalize(_randn((clusters, n, 3), generator, device), dim=-1)
    u = _rand((clusters, n), generator, device)
    radius = torch.empty_like(u)
    start = 0
    for count, low, high in zip(counts, edges, edges[1:]):
        part = u[:, start:start + count]
        radius[:, start:start + count] = (low ** 3 + (high ** 3 - low ** 3) * part).pow(1.0 / 3.0)
        start += count
    order = torch.argsort(_rand((clusters, n), generator, device), dim=1)   # bands mixed along the rows
    return torch.gather(direction * radius.unsqueeze(-1), 1, order.unsqueeze(-1).expand(-1, -1, 3))


@dataclass
class FieldBatch:
    codes: Tensor    # B×D unit codes
    cluster: Tensor  # B long, cluster index of each code
    x: Tensor        # B×N×3 volume samples; identical for codes of one cluster; x[:, 0] == 0 (origin)
    f: Tensor        # B×N    field at x                (in the autograd graph)
    g: Tensor        # B×N×3  spatial gradient at x      (create_graph=True)
    xp: Tensor       # B×N×3  one-step surface guess x − f·g
    fp: Tensor       # B×N    field at xp               (in the autograd graph)
    mod: Any = None  # decoder modulation of `codes` (in the graph), for further queries
    fp_weight: Optional[Tensor] = None  # B×N weights of the consistency term (None = all 1)
    f_move: Optional[Tensor] = None     # B×N, equal to f in value; its gradient also reaches the field at xp
    # B×N weights that turn a plain mean over the samples into a mean over the box (volume
    # integrals: volume, centroid, covariance). None = the samples are uniform in the box.
    volume_weight: Optional[Tensor] = None


def sample_points(
    cfg: DataFreeConfig,
    clusters: int,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
    extra: Optional[Tensor] = None,
) -> Tensor:
    """``C×N×3`` volume samples, uniform in the box (or in radial shells, per
    ``cfg.point_sampling``), row 0 the origin.

    ``extra`` (``C×H×3``, NaN-padded) holds hard negatives: its valid rows are
    written over the last rows of each cluster's samples (clamped to the box).
    Rows of ``extra`` that are NaN leave the uniform sample in place. At most
    ``N - 1`` extra points are used, so the origin is never overwritten.
    """
    device = _device_of(device, generator)
    n = cfg.points
    if str(getattr(cfg, "point_sampling", "uniform")) == "shells":
        x = shell_points(cfg, clusters, generator, device)
    else:
        x = _rand((clusters, n, 3), generator, device) * 2.0 - 1.0
    x[:, 0] = 0.0
    if extra is not None and extra.shape[1] > 0 and n > 1:
        h = min(extra.shape[1], n - 1)
        e = extra[:, :h].to(device=device, dtype=x.dtype)
        valid = torch.isfinite(e).all(dim=-1, keepdim=True)
        tail = x[:, n - h:]
        x[:, n - h:] = torch.where(valid, e.clamp(-1.0, 1.0), tail)
    return x


def field_and_grad(decoder, mod, x: Tensor, create_graph: bool = True) -> Tuple[Tensor, Tensor]:
    """Field ``f`` (B×N) and spatial gradient ``g = ∇_x f`` (B×N×3) at ``x``.

    ``f`` stays in the autograd graph of the decoder parameters. With
    ``create_graph=True`` so does ``g``, so a loss on the gradient can be
    back-propagated; otherwise ``g`` is detached. Works under ``torch.no_grad``.
    If ``x`` does not require grad, a detached copy that does is used.
    """
    with torch.enable_grad():
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)
        f = decoder.query(mod, x)
        # Each f[b, n] depends only on x[b, n], so one backward pass of the sum
        # gives every per-point gradient.
        (g,) = torch.autograd.grad(f.sum(), x, create_graph=create_graph, retain_graph=True)
    return f, g


def evaluate_field(
    decoder,
    codes: Tensor,
    cluster: Tensor,
    x: Tensor,
    cfg: DataFreeConfig,
    create_graph: bool = True,
) -> FieldBatch:
    """Evaluate everything the losses and the graph reuse, in two network passes.

    ``x`` is ``C×N×3`` (one row per cluster, as returned by ``sample_points``;
    expanded with ``x[cluster]``) or already ``B×N×3`` (one row per code). A
    first dimension equal to ``cluster.max() + 1`` is read as per cluster.
    """
    b = codes.shape[0]
    cluster = cluster.to(device=codes.device, dtype=torch.long)
    num_clusters = int(cluster.max()) + 1 if b > 0 else 0
    if x.shape[0] == num_clusters:
        x = x[cluster]
    elif x.shape[0] != b:
        raise ValueError(
            f"x has {x.shape[0]} rows; expected {num_clusters} (per cluster) or {b} (per code)"
        )
    x = x.detach()

    mod = decoder.prepare(codes)
    f, g = field_and_grad(decoder, mod, x, create_graph=create_graph)
    direction = g.detach() if cfg.consistency_detach_direction else g
    xp = x - f[..., None] * direction
    weight = None
    moving = None
    scale = float(getattr(cfg, "consistency_surface_scale", 1.0))
    fixed = bool(getattr(cfg, "consistency_fixed_surface", False)) or scale <= 0.0
    if bool(getattr(cfg, "field_moves_surface", False)):
        # One pass at the landing points with the weights attached and the positions detached.
        # f_move carries f's value, and its gradient reaches both the field at x and the field at
        # the landing point, so a push on the value at x moves the piece of surface x belongs to.
        # The landing value fp is rebuilt to first order from that pass and its spatial slope:
        # value unchanged, gradient through the starting point always, and through the landing
        # point only when the surface is not held fixed.
        with torch.enable_grad():
            at = xp.detach().requires_grad_(True)
            landed = decoder.query(mod, at)
            (slope,) = torch.autograd.grad(landed.sum(), at, retain_graph=True)
        moving = f + (landed - landed.detach())
        through_start = (slope.detach() * (xp - xp.detach())).sum(dim=-1)
        fp = (landed.detach() if fixed else landed) + through_start
        depth = float(getattr(cfg, "consistency_min_depth", 0.0))
        if fixed and depth > 0:
            weight = (f.detach().abs() >= depth).to(f.dtype)
    elif fixed:
        # The landing value with the network's weights held fixed: its gradient reaches the
        # parameters only through xp (the field and gradient at the starting point), so the
        # consistency term cannot move the surface the step lands on.
        frozen = {name: value.detach() for name, value in decoder.named_parameters()}
        fp = torch.func.functional_call(decoder, frozen, (codes.detach(), xp))
        depth = float(getattr(cfg, "consistency_min_depth", 0.0))
        if depth > 0:
            weight = (f.detach().abs() >= depth).to(f.dtype)
    elif scale < 1.0:
        # The dial: same values, but the gradient that reaches the weights through the landing
        # point is multiplied by `scale` (the route through the starting point keeps full strength).
        partial = {name: scale * value + (1.0 - scale) * value.detach()
                   for name, value in decoder.named_parameters()}
        fp = torch.func.functional_call(decoder, partial, (codes.detach(), xp))
        depth = float(getattr(cfg, "consistency_min_depth", 0.0))
        if depth > 0:
            weight = (f.detach().abs() >= depth).to(f.dtype)
    else:
        fp = decoder.query(mod, xp)
    return FieldBatch(
        codes=torch.nn.functional.normalize(codes, dim=-1),
        cluster=cluster,
        x=x,
        f=f,
        g=g,
        xp=xp,
        fp=fp,
        mod=mod,
        fp_weight=weight,
        f_move=moving,
        volume_weight=(shell_volume_weight(x, cfg).expand(f.shape)
                       if str(getattr(cfg, "point_sampling", "uniform")) == "shells" else None),
    )
