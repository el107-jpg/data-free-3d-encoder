"""How flat, bent or creased are the surfaces a model decodes at random codes?

usage: sharpness.py <out.json> <label>=<checkpoint> [<label>=<checkpoint> ...]

For 64 fresh random codes (same codes for every model), 3,000 points per code are slid onto the surface with
Newton steps. At each, the unit normal is compared with the normal 0.02 away along a random direction in the
surface (the partner is slid back onto the surface too). bend = |change of normal| / distance: 1 / radius on a
ball, 0 on a plane, about 70 across a right-angled edge. Reported per model: median bend, share of surface
that is flat (bend under 1, radius over 1), gently bent (1 to 9), tight (over 9, radius under 0.11), and
the training term's mean cost bend^2 / (bend^2 + 9). Independent of the training code's point selection.
"""
import json
import sys

import numpy as np
import torch

from arm_planner.data_free.train import Trainer

CODES, POINTS, STEP, CHUNK = 64, 3000, 0.02, 16


def slide(model, codes, x, steps):
    for _ in range(steps):
        x = x.detach().requires_grad_(True)
        f = model(codes, x)
        (g,) = torch.autograd.grad(f.sum(), x)
        move = f.detach().unsqueeze(-1) * g / g.square().sum(-1, keepdim=True).clamp_min(1e-4)
        x = (x - move.clamp(-0.1, 0.1)).detach()
    x = x.requires_grad_(True)
    f = model(codes, x)
    (g,) = torch.autograd.grad(f.sum(), x)
    return x.detach(), f.detach(), g.detach()


out = {}
for item in sys.argv[2:]:
    label, path = item.split("=", 1)
    trainer = Trainer.from_checkpoint(path, device="cuda", tensorboard=False)
    cfg, model = trainer.cfg, trainer.model.eval()
    gen = torch.Generator().manual_seed(2718)
    codes = torch.nn.functional.normalize(torch.randn(CODES, int(cfg.latent_dim), generator=gen), dim=-1).cuda()
    bends = []
    for s in range(0, CODES, CHUNK):
        c = codes[s:s + CHUNK]
        x = (torch.rand(c.shape[0], POINTS, 3, generator=gen) * 2 - 1).cuda()
        p, fp, gp = slide(model, c, x, 8)
        n = gp / gp.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        t = torch.randn(p.shape, generator=gen).cuda()
        t = t - (t * n).sum(-1, keepdim=True) * n
        t = t / t.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        q, fq, gq = slide(model, c, p + STEP * t, 3)
        m = gq / gq.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        gap = (q - p).norm(dim=-1)
        ok = ((fp.abs() < 2e-3) & (fq.abs() < 2e-3) & (p.norm(dim=-1) < float(cfg.bound_radius))
              & (gap > 0.5 * STEP) & (gap < 2.0 * STEP) & (gp.norm(dim=-1) > 0.2) & (gq.norm(dim=-1) > 0.2))
        bends.append(((n - m).norm(dim=-1) / gap.clamp_min(1e-6))[ok].cpu().numpy())
    b = np.concatenate(bends)
    out[label] = {
        "checkpoint": path, "update": int(trainer.step_count), "pairs": int(b.size),
        "median_bend": float(np.median(b)), "flat": float((b < 1).mean()), "gentle": float(((b >= 1) & (b <= 9)).mean()),
        "tight": float((b > 9).mean()), "cost": float((b ** 2 / (b ** 2 + 9.0)).mean()),
        "histogram": np.histogram(np.log10(b.clip(1e-2, 1e3)), bins=np.linspace(-2, 3, 51))[0].tolist(),
    }
    r = out[label]
    print(f"{label} (update {r['update']}, {r['pairs']} surface pairs): median bend {r['median_bend']:.2f} (radius {1 / r['median_bend']:.2f}); "
          f"flat {r['flat']:.1%}, gently bent {r['gentle']:.1%}, tight {r['tight']:.1%}; mean cost {r['cost']:.3f}", flush=True)
    del trainer, model
json.dump(out, open(sys.argv[1], "w"), indent=1)
