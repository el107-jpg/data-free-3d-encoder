"""Calibrate the weak-direction pairs on a saved model.

1. Usage spectrum in the trainer's own measure (RMS field change per unit chord per code-space direction).
2. Pair rates (distance / (slope * chord), pose removed) for three kinds of partner:
   a. used part kept, unused part turned (what training uses),
   b. a plain step along an unused axis (shows the rescaling leak),
   c. a plain step along the most-used axis.
"""
import math
import sys

import torch

from arm_planner.data_free import losses as L
from arm_planner.data_free.field import evaluate_field, sample_points
from arm_planner.data_free.train import Trainer

ckpt = sys.argv[1]
floor = float(sys.argv[2]) if len(sys.argv) > 2 else 0.02
trainer = Trainer.from_checkpoint(ckpt, device="cuda", tensorboard=False, overrides=[f"weak_floor={floor}"])
cfg, model = trainer.cfg, trainer.model
usage = trainer.measure_usage()
lam, vec = torch.linalg.eigh(usage)
lam, vec = lam.flip(0).clamp_min(0), vec.flip(1)
print(f"update {trainer.step_count}; RMS field change per unit chord, by code-space direction (sorted):")
print(" ".join(f"{v:.4f}" for v in lam.sqrt().tolist()))
print("info:", {k: round(v, 4) for k, v in trainer._usage_info.items()}, "| unused directions:", trainer._weak_basis.shape[1])

gen = torch.Generator().manual_seed(3)
n = 12
chord = float(cfg.weak_chord)
base = torch.nn.functional.normalize(torch.randn(n, 32, generator=gen), dim=-1).cuda()
basis = trainer._weak_basis


def turned(base):
    inside = (base @ basis) @ basis.T
    size = inside.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    noise = torch.randn((base.shape[0], basis.shape[1]), generator=gen).cuda() @ basis.T
    turn = torch.nn.functional.normalize(noise - (noise * inside).sum(-1, keepdim=True) * inside / size.square(), dim=-1)
    angle = 2.0 * torch.asin((0.5 * chord / size).clamp(max=1.0))
    return torch.nn.functional.normalize((base - inside) + torch.cos(angle) * inside + torch.sin(angle) * size * turn, dim=-1)


def stepped(base, axis):
    d = axis[None] - (base @ axis)[:, None] * base
    d = torch.nn.functional.normalize(d, dim=-1)
    theta = 2.0 * math.asin(0.5 * chord)
    return torch.nn.functional.normalize(math.cos(theta) * base + math.sin(theta) * d, dim=-1)


x = sample_points(cfg, 1, gen, "cuda")
first, second = torch.arange(n).cuda(), torch.arange(n, 2 * n).cuda()
for name, partner in (("used part kept, unused part turned", turned(base)),
                      ("plain step along an unused axis", stepped(base, vec[:, -1].float().cuda())),
                      ("plain step along the 12th axis", stepped(base, vec[:, 11].float().cuda())),
                      ("plain step along the 9th axis", stepped(base, vec[:, 8].float().cuda())),
                      ("plain step along the most-used axis", stepped(base, vec[:, 0].float().cuda()))):
    codes = torch.cat([base, partner])
    fb = evaluate_field(model, codes, torch.zeros(2 * n, dtype=torch.long, device="cuda"), x, cfg, create_graph=False)
    lengths = (base - partner).norm(dim=-1)
    f = fb.f[:, 1:]
    xx, g = fb.x[:, 1:].detach(), fb.g[:, 1:].detach()
    mask = (xx[:n].square().sum(-1) < cfg.bound_radius ** 2).float()
    delta = f[:n] - f[n:]
    delta = L._remove_pose(delta, xx[:n], g[:n], (0.5 * (f[:n] + f[n:])).detach(), mask)
    dist = (delta.abs() * mask).sum(1) / mask.sum(1)
    rate = dist / (cfg.diversity_slope * lengths)
    print(f"{name:38s} chord {float(lengths.mean()):.3f}  rate mean {float(rate.mean()):.5f}  min {float(rate.min()):.5f}  max {float(rate.max()):.5f}")
    del fb
