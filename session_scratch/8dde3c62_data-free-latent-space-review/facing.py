"""How does the surface at random codes face relative to the centre, and where is the slope wrong?

usage: facing.py <checkpoint> [<checkpoint> ...]

Works for any decoder form (it uses the decoder's own field). 32 random codes, points within 0.03 of the
surface. facing = cosine between the outward normal and the direction away from the centre: +1 faces outward,
0 side-on, -1 faces the centre. A star-shaped body (every ray from the centre crosses the surface once) has
no surface with facing below 0.
"""
import sys

import numpy as np
import torch

from arm_planner.data_free.train import Trainer

for path in sys.argv[1:]:
    trainer = Trainer.from_checkpoint(path, device="cuda", tensorboard=False)
    cfg, model = trainer.cfg, trainer.model.eval()
    gen = torch.Generator().manual_seed(77)
    codes = torch.nn.functional.normalize(torch.randn(32, cfg.latent_dim, generator=gen), dim=-1).cuda()
    err, rho, slope, facing = [], [], [], []
    for k in range(32):
        mod = model.prepare(codes[k:k + 1])
        x = (torch.rand(1, 200000, 3, generator=gen) * 2 - 1).cuda()
        with torch.no_grad():
            f = model.query(mod, x)
        x = x[:, f[0].abs() < 0.03][:, :6000].requires_grad_(True)
        with torch.enable_grad():
            f = model.query(mod, x)
            (g,) = torch.autograd.grad(f.sum(), x)
        r = x.norm(dim=-1)
        size = g[0].norm(dim=-1)
        facing.append(((-g[0] / size[:, None].clamp_min(1e-6)) * (x[0] / r[0][:, None])).sum(-1).detach().cpu().numpy())
        rho.append(r[0].detach().cpu().numpy()); slope.append(size.cpu().numpy()); err.append((size - 1).abs().cpu().numpy())
    err, rho, slope, facing = (np.concatenate(v) for v in (err, rho, slope, facing))
    print(f"\n{path.split('/')[-2][:44]} (update {trainer.step_count}); {err.size} points within 0.03 of the surface; "
          f"slope error {err.mean():.3f}, median slope {np.median(slope):.2f}; surface below facing 0: {(facing < 0).mean():.3f}")
    for lo, hi, name in ((0.9, 1.01, "faces outward"), (0.5, 0.9, "tilted"), (0.0, 0.5, "nearly side-on"),
                         (-0.5, 0.0, "past side-on"), (-1.01, -0.5, "faces the centre")):
        m = (facing >= lo) & (facing < hi)
        if m.sum() >= 20:
            print(f"    {name:17s}: share {m.mean():.3f}, mean radius {rho[m].mean():.2f}, slope error {err[m].mean():.3f}, median slope {np.median(slope[m]):.2f}")
        else:
            print(f"    {name:17s}: share {m.mean():.4f}")
