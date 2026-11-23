"""Where near the surface is the slope wrong, and is it where the bounded correction is saturated?

usage: saturation.py <checkpoint> [<checkpoint> ...]

f = (R - rho) + c with c = A tanh(raw / A) (A = correction_bound for c >= 0, correction_bound_inner for c < 0).
For 32 random codes, points within 0.03 of the surface (from 200,000 uniform samples per code). For each point:
saturation s = |c| / bound (0 = linear part of tanh, 1 = flat), the factor 1 / (1 - s^2) by which the raw
network's slope must be larger to give the same slope of f, radius rho, and the slope error | |grad f| - 1 |.
"""
import sys

import numpy as np
import torch

from arm_planner.data_free.train import Trainer

BINS = [0.0, 0.3, 0.6, 0.8, 0.9, 0.95, 1.0]
for path in sys.argv[1:]:
    trainer = Trainer.from_checkpoint(path, device="cuda", tensorboard=False)
    cfg, model = trainer.cfg, trainer.model.eval()
    a, b = float(model.correction_bound), float(model.correction_inner)
    gen = torch.Generator().manual_seed(77)
    codes = torch.nn.functional.normalize(torch.randn(32, cfg.latent_dim, generator=gen), dim=-1).cuda()
    sat, err, rho, slope, facing = [], [], [], [], []
    for k in range(32):
        mod = model.prepare(codes[k:k + 1])
        x = (torch.rand(1, 200000, 3, generator=gen) * 2 - 1).cuda()
        with torch.no_grad():
            f = model.query(mod, x)
        x = x[:, f[0].abs() < 0.03][:, :6000].requires_grad_(True)
        with torch.enable_grad():
            raw = model._net(mod, x)
            c = torch.where(raw >= 0, a * torch.tanh(raw / a), b * torch.tanh(raw / b))
            r = x.norm(dim=-1)
            f = model.base_radius - r + c
            (g,) = torch.autograd.grad(f.sum(), x)
        s = torch.where(c >= 0, c / a, -c / b).detach()
        sat.append(s[0].cpu().numpy()); rho.append(r[0].detach().cpu().numpy())
        facing.append(((-g[0] / g[0].norm(dim=-1, keepdim=True)) * (x[0] / r[0][:, None])).sum(-1).detach().cpu().numpy())
        slope.append(g[0].norm(dim=-1).cpu().numpy()); err.append((g[0].norm(dim=-1) - 1).abs().cpu().numpy())
    sat, err, rho, slope, facing = (np.concatenate(v) for v in (sat, err, rho, slope, facing))
    print(f"\n{path.split('/')[-2][:40]} (update {trainer.step_count}); {sat.size} points within 0.03 of the surface, 32 codes; "
          f"overall slope error {err.mean():.3f} (median slope {np.median(slope):.2f})")
    print(f"  {'saturation':>12s} {'share of surface':>17s} {'stretch needed':>15s} {'mean radius':>12s} {'slope error':>12s} {'median slope':>13s}")
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (sat >= lo) & (sat < hi)
        if m.sum() < 20:
            print(f"  {lo:5.2f}-{hi:4.2f}  {m.mean():17.3f}"); continue
        print(f"  {lo:5.2f}-{hi:4.2f}  {m.mean():17.3f} {np.mean(1 / (1 - sat[m] ** 2)):15.1f} {rho[m].mean():12.2f} "
              f"{err[m].mean():12.3f} {np.median(slope[m]):13.2f}")
    print("  by how the surface faces (cosine between the outward normal and the direction away from the centre):")
    for lo, hi, name in ((0.9, 1.01, "faces outward"), (0.5, 0.9, "tilted"), (-0.5, 0.5, "side-on"), (-1.01, -0.5, "faces the centre")):
        m = (facing >= lo) & (facing < hi)
        if m.sum() >= 20:
            print(f"    {name:17s} ({lo:+.1f} to {hi:+.1f}): share {m.mean():.3f}, mean radius {rho[m].mean():.2f}, slope error {err[m].mean():.3f}, median slope {np.median(slope[m]):.2f}")
    print("  by radius:")
    for lo, hi in ((0.0, 0.3), (0.3, 0.5), (0.5, 0.65), (0.65, 0.75), (0.75, 0.85)):
        m = (rho >= lo) & (rho < hi)
        if m.sum() >= 20:
            print(f"    radius {lo:.2f}-{hi:.2f}: share {m.mean():.3f}, mean saturation {sat[m].mean():.2f}, slope error {err[m].mean():.3f}, median slope {np.median(slope[m]):.2f}")
