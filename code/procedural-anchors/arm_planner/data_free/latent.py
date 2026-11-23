"""Latent codes on the unit sphere: sampling, chord distance, slerp, clusters, tangents."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor


def _device_of(device, generator: Optional[torch.Generator]) -> torch.device:
    if device is not None:
        return torch.device(device)
    if generator is not None:
        return generator.device
    return torch.device("cpu")


def _draw(fn, shape: Tuple[int, ...], generator: Optional[torch.Generator], device) -> Tensor:
    device = torch.device(device)
    if generator is None or generator.device.type == device.type:
        return fn(shape, generator=generator, device=device)
    # A generator can only fill tensors on its own device.
    return fn(shape, generator=generator, device=generator.device).to(device)


def _rand(shape, generator, device) -> Tensor:
    return _draw(torch.rand, tuple(shape), generator, device)


def _randn(shape, generator, device) -> Tensor:
    return _draw(torch.randn, tuple(shape), generator, device)


def sample_sphere(
    n: int,
    dim: int,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    """``n×dim`` unit vectors, uniform on the sphere (normalised Gaussian)."""
    return F.normalize(_randn((n, dim), generator, _device_of(device, generator)), dim=-1)


def chord(a: Tensor, b: Tensor) -> Tensor:
    """Chord length ``sqrt(2 - 2 cos)`` between codes, broadcasting over leading dims.

    ``cos`` is the cosine similarity, so the inputs need not be exactly unit.
    The gradient is zero (not infinite or NaN) where the codes coincide.
    """
    cos = (F.normalize(a, dim=-1) * F.normalize(b, dim=-1)).sum(-1)
    d2 = (2.0 - 2.0 * cos).clamp_min(0.0)
    positive = d2 > 1e-12
    return torch.where(positive, torch.sqrt(torch.where(positive, d2, torch.ones_like(d2))), torch.zeros_like(d2))


def slerp(a: Tensor, b: Tensor, t: Union[Tensor, float]) -> Tensor:
    """Point at fraction ``t`` of the great-circle arc from ``a`` to ``b`` (unit output).

    ``t`` is a float or a tensor broadcastable against ``a[..., 0]``. Nearly
    parallel codes fall back to a normalised straight-line blend. Exactly
    opposite codes have no unique arc; the result is then arbitrary but finite.
    """
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    t = torch.as_tensor(t, dtype=a.dtype, device=a.device)[..., None]
    cos = (a * b).sum(-1, keepdim=True).clamp(-1.0, 1.0)
    omega = torch.acos(cos)
    sin = torch.sin(omega)
    small = sin < 1e-4
    safe = torch.where(small, torch.ones_like(sin), sin)
    wa = torch.where(small, 1.0 - t, torch.sin((1.0 - t) * omega) / safe)
    wb = torch.where(small, t, torch.sin(t * omega) / safe)
    return F.normalize(wa * a + wb * b, dim=-1)


def sample_cluster(
    center: Tensor,
    k: int,
    eps: Union[Tensor, float],
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """``k`` unit codes within chord distance ``eps`` of ``center``; the first is ``center``.

    ``center`` is ``D`` (returns ``k×D``) or ``C×D`` with ``eps`` a float or a
    ``C`` tensor (returns ``C×k×D``). The other ``k - 1`` codes lie in uniformly
    random directions at chord radii stratified over ``(0, eps]`` (one per equal
    slice), so distances to the centre and between members cover a range.
    """
    single = center.dim() == 1
    c = F.normalize(center.reshape(-1, center.shape[-1]), dim=-1)  # C×D
    n, d = c.shape
    eps = torch.as_tensor(eps, dtype=c.dtype, device=c.device).reshape(-1, 1).expand(n, 1).clamp(0.0, 2.0)
    if k <= 1:
        out = c[:, None, :]
        return out[0] if single else out

    m = k - 1
    # Random unit tangent directions at the centre.
    u = _randn((n, m, d), generator, c.device).to(c.dtype)
    u = F.normalize(u - (u * c[:, None, :]).sum(-1, keepdim=True) * c[:, None, :], dim=-1)
    # Stratified chord radii, in a random order along the member index.
    strata = torch.arange(m, device=c.device, dtype=c.dtype).expand(n, m)
    frac = (strata + _rand((n, m), generator, c.device).to(c.dtype)) / m
    order = _rand((n, m), generator, c.device).argsort(dim=1)
    radius = eps * torch.gather(frac, 1, order)
    theta = 2.0 * torch.asin((radius / 2.0).clamp(max=1.0))[..., None]
    members = F.normalize(torch.cos(theta) * c[:, None, :] + torch.sin(theta) * u, dim=-1)
    out = torch.cat([c[:, None, :], members], dim=1)
    return out[0] if single else out


def tangent_basis(z: Tensor) -> Tensor:
    """``(dim-1)×dim`` orthonormal basis of the tangent space of the sphere at ``z``."""
    z = F.normalize(z.reshape(-1), dim=0)
    d = z.shape[0]
    a = torch.cat([z[:, None], torch.eye(d, dtype=z.dtype, device=z.device)], dim=1)
    q, _ = torch.linalg.qr(a)  # first column is ±z, the rest span its complement
    basis = q[:, 1:].T
    # One clean-up pass keeps the rows orthogonal to z to rounding error.
    return F.normalize(basis - (basis @ z)[:, None] * z, dim=-1)
