"""Distil a trained decoder into another decoder form (e.g. one without the fixed sphere base).

The student is trained to reproduce the teacher's field, value and spatial slope, at fresh random codes (and
the anchor codes, when there are any) every update: half of each code's points uniform in the box, half
the points closest to the teacher's surface out of a larger uniform draw. Nothing else is in the loss.

Typical use (scripts/distill_data_free.py): the student is the teacher's own checkpoint loaded with changed
settings (sphere_base=False, ...), so it starts from the teacher's weights and only has to learn the part of
the field the new form no longer provides by construction; the student's trainer state (step, multipliers,
anchor codes) is the teacher's, and its optimiser state is cleared before training continues.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional

import torch
from torch import Tensor

from .field import field_and_grad


def _points(teacher, codes: Tensor, count: int, generator: torch.Generator, device) -> Tensor:
    """Per code: count // 2 uniform points in the box and count // 2 near the teacher's surface."""
    b = codes.shape[0]
    half = count // 2
    pool = torch.rand(b, 4 * count, 3, generator=generator, device=device) * 2 - 1
    with torch.no_grad():
        f = teacher(codes, pool)
    near = f.abs().topk(half, dim=1, largest=False).indices
    picked = torch.gather(pool, 1, near[..., None].expand(-1, -1, 3))
    rest = torch.rand(b, count - half, 3, generator=generator, device=device) * 2 - 1
    return torch.cat([picked, rest], dim=1)


def distill(student, teacher, latent_dim: int, steps: int, *, anchor_codes: Optional[Tensor] = None,
            codes_per_step: int = 24, anchors_per_step: int = 8, points: int = 2048, lr: float = 3e-4,
            slope_weight: float = 0.1, generator: Optional[torch.Generator] = None,
            log: Optional[Callable[[Dict[str, float]], None]] = None, log_every: int = 100) -> List[Dict[str, float]]:
    """Train ``student`` (all its parameters) to reproduce ``teacher``'s field; returns the logged records.

    Loss per update: mean |f_s - f_t| + slope_weight * mean |grad f_s - grad f_t|^2. Adam at ``lr`` with a
    100-update warm-up and a cosine to a tenth. The teacher is frozen and left in eval mode.
    """
    device = next(student.parameters()).device
    gen = generator if generator is not None else torch.Generator(device=device).manual_seed(0)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student.train()
    params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    warm = min(100, max(1, steps // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(steps - warm, 1))))
    history: List[Dict[str, float]] = []
    for step in range(steps):
        codes = torch.nn.functional.normalize(torch.randn(codes_per_step, latent_dim, generator=gen, device=device), dim=-1)
        if anchor_codes is not None and anchors_per_step > 0:
            pick = torch.randint(0, anchor_codes.shape[0], (anchors_per_step,), generator=gen, device=device)
            codes = torch.cat([codes, torch.nn.functional.normalize(anchor_codes.detach()[pick], dim=-1)])
        x = _points(teacher, codes, points, gen, device)
        f_t, g_t = field_and_grad(teacher, teacher.prepare(codes), x, create_graph=False)
        f_s, g_s = field_and_grad(student, student.prepare(codes), x, create_graph=True)
        value = (f_s - f_t.detach()).abs().mean()
        slope = (g_s - g_t.detach()).square().sum(-1).mean()
        loss = value + slope_weight * slope
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 10.0)
        opt.step()
        sched.step()
        if step % log_every == 0 or step == steps - 1:
            with torch.no_grad():
                near = f_t.abs() < 0.05
                record = {"step": step, "value_error": float(value), "slope_mismatch": float(slope),
                          "value_error_near_surface": float((f_s - f_t).abs()[near].mean()) if bool(near.any()) else float("nan"),
                          "sign_disagreement": float(((f_s > 0) != (f_t > 0)).float().mean()), "lr": float(sched.get_last_lr()[0])}
            history.append(record)
            if log is not None:
                log(record)
    student.eval()
    return history


@torch.no_grad()
def agreement(student, teacher, latent_dim: int, codes: int = 64, points: int = 20000,
              generator: Optional[torch.Generator] = None) -> Dict[str, float]:
    """How closely the student reproduces the teacher at fresh random codes: mean |f_s - f_t| over the ball and
    within 0.05 of the teacher's surface, the share of points where the inside/outside sign differs, and the
    mean overlap (intersection over union) of the two solids."""
    device = next(student.parameters()).device
    gen = generator if generator is not None else torch.Generator(device=device).manual_seed(1)
    z = torch.nn.functional.normalize(torch.randn(codes, latent_dim, generator=gen, device=device), dim=-1)
    x = torch.randn(points, 3, generator=gen, device=device)
    x = (x / x.norm(dim=-1, keepdim=True) * 0.9 * torch.rand(points, 1, generator=gen, device=device) ** (1 / 3))
    errs, near_errs, signs, ious = [], [], [], []
    for s0 in range(0, codes, 16):
        c = z[s0:s0 + 16]
        xs = x[None].expand(c.shape[0], -1, -1)
        f_s, f_t = student(c, xs), teacher(c, xs)
        near = f_t.abs() < 0.05
        errs.append((f_s - f_t).abs().mean(1))
        near_errs.append(torch.stack([(f_s[i] - f_t[i]).abs()[near[i]].mean() for i in range(c.shape[0])]))
        signs.append(((f_s > 0) != (f_t > 0)).float().mean(1))
        both = ((f_s > 0) & (f_t > 0)).sum(1).float()
        either = ((f_s > 0) | (f_t > 0)).sum(1).float().clamp_min(1)
        ious.append(both / either)
    return {"value_error": float(torch.cat(errs).mean()), "value_error_near_surface": float(torch.cat(near_errs).nanmean()),
            "sign_disagreement": float(torch.cat(signs).mean()), "overlap_mean": float(torch.cat(ious).mean()),
            "overlap_worst": float(torch.cat(ious).min())}
