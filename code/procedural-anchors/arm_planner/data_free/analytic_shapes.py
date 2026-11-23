"""Analytic signed distance fields (positive inside) for tests and benchmarks of the graph.

Each shape is a callable ``points (..., 3) -> values (...)``. Unions use
``max`` and intersections ``min``, which keep the sign exact everywhere and the
distance exact on the side noted in each constructor.
"""

from __future__ import annotations

from typing import Callable, Dict, Sequence, Tuple

import torch
from torch import Tensor

Field = Callable[[Tensor], Tensor]


def _norm(v: Tensor) -> Tensor:
    return torch.sqrt(v.square().sum(dim=-1) + 1e-24)


def sphere(center: Sequence[float], radius: float) -> Field:
    def field(p: Tensor) -> Tensor:
        return radius - _norm(p - p.new_tensor(center))
    return field


def box(center: Sequence[float], half: Sequence[float]) -> Field:
    def field(p: Tensor) -> Tensor:
        q = (p - p.new_tensor(center)).abs() - p.new_tensor(half)
        outside = _norm(q.clamp_min(0.0))
        inside = q.amax(dim=-1).clamp_max(0.0)
        return -(outside + inside)
    return field


def torus(major: float, minor: float) -> Field:
    def field(p: Tensor) -> Tensor:
        ring = _norm(p[..., :2]) - major
        return minor - torch.sqrt(ring.square() + p[..., 2].square() + 1e-24)
    return field


def capsule(a: Sequence[float], b: Sequence[float], radius: float) -> Field:
    def field(p: Tensor) -> Tensor:
        pa, pb = p.new_tensor(a), p.new_tensor(b)
        ab = pb - pa
        t = (((p - pa) * ab).sum(dim=-1) / ab.square().sum()).clamp(0.0, 1.0)
        return radius - _norm(p - pa - t.unsqueeze(-1) * ab)
    return field


def union(*fields: Field) -> Field:
    """Exact distance outside, a lower bound inside."""
    def field(p: Tensor) -> Tensor:
        return torch.stack([f(p) for f in fields], dim=0).amax(dim=0)
    return field


def intersection(*fields: Field) -> Field:
    """Exact distance inside, a lower bound (in magnitude) outside."""
    def field(p: Tensor) -> Tensor:
        return torch.stack([f(p) for f in fields], dim=0).amin(dim=0)
    return field


def complement(f: Field) -> Field:
    return lambda p: -f(p)


def shapes() -> Dict[str, Field]:
    """The named test shapes."""
    # 0.05 wide, cut from the +x face to x = -0.25 (the box spans x in [-0.6, 0.6])
    slot = box([0.2, 0.15, 0.0], [0.45, 0.025, 1.0])
    return {
        "sphere": sphere([0, 0, 0], 0.5),
        "two_spheres": union(sphere([-0.35, 0, 0], 0.4), sphere([0.55, 0, 0], 0.3)),       # gap 0.2
        "two_spheres_near": union(sphere([-0.3, 0, 0], 0.4), sphere([0.45, 0, 0], 0.3)),   # gap 0.05
        "shell": intersection(sphere([0, 0, 0], 0.8), complement(sphere([0, 0, 0], 0.4))),
        "torus": torus(0.55, 0.25),
        "slotted_box": intersection(box([0, 0, 0], [0.6, 0.4, 0.4]), complement(slot)),
        "plate": box([0, 0, 0], [0.6, 0.6, 0.02]),                                         # 0.04 thick
        "box": box([0, 0, 0], [0.5, 0.4, 0.3]),
        "dumbbell": union(sphere([-0.45, 0, 0], 0.3), sphere([0.45, 0, 0], 0.3),
                          capsule([-0.45, 0, 0], [0.45, 0, 0], 0.15)),
    }


def sample_inputs(
    fields: Sequence[Field], n: int = 2048, seed: int = 0, device="cpu", noise: float = 0.0,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """``(x, f, g, xp, fp)`` as a training step would have them, one row per field.

    Uniform box samples with ``x[:, 0]`` the origin, exact gradients by
    autograd, ``xp = x − f·g``. ``noise`` adds a random value error of that
    size to ``f`` and ``fp`` (signs kept), to imitate an imperfect field.
    """
    gen = torch.Generator().manual_seed(seed)
    xs, fs, gs, xps, fps = [], [], [], [], []
    for field in fields:
        x = torch.rand(n, 3, generator=gen) * 2.0 - 1.0
        x[0] = 0.0
        x = x.to(device).requires_grad_(True)
        f = field(x)
        (g,) = torch.autograd.grad(f.sum(), x)
        x, f = x.detach(), f.detach()
        xp = x - f.unsqueeze(-1) * g
        fp = field(xp)
        if noise > 0:
            scale = 1.0 + noise * (torch.rand(2, n, generator=gen).to(device) * 2 - 1)
            f, fp = f * scale[0], fp * scale[1]
        xs.append(x), fs.append(f), gs.append(g), xps.append(xp), fps.append(fp)
    return (torch.stack(xs), torch.stack(fs), torch.stack(gs), torch.stack(xps), torch.stack(fps))
