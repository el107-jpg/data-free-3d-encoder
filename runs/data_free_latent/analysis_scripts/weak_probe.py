"""Did the directions that were unused at update 2,500 come alive?

The unused directions are fixed once, from the 2,500-update model, and reused for every model so
the comparison is like for like. For each model:
 - pair rate (pose-removed field distance / (slope * chord)) for codes whose used part is identical
   and whose unused part is turned by chords 0.1 .. 1.0;
 - how far the surface moves between such codes (RMS, pose not removed, at the base shape's surface);
 - the trainer's usage spectrum.
Also a picture: one slice of a base code and of three codes with the same used part and a
strongly turned unused part, for each model.

usage: weak_probe.py <reference ckpt> <label> <ckpt> [<label> <ckpt> ...]
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from arm_planner.data_free import losses as L
from arm_planner.data_free.field import evaluate_field, sample_points
from arm_planner.data_free.train import Trainer

ref_ckpt = sys.argv[1]
models = list(zip(sys.argv[2::2], sys.argv[3::2]))
tag = sys.argv[2].replace(" ", "_")
ref = Trainer.from_checkpoint(ref_ckpt, device="cuda", tensorboard=False, overrides=["weak_floor=0.04", "weak_probe_codes=32"])
ref.measure_usage()
basis = ref._weak_basis.clone()
print(f"unused directions fixed from {ref_ckpt.split('/')[-2]}/{ref_ckpt.split('/')[-1]}: {basis.shape[1]} of 32")
del ref
torch.cuda.empty_cache()

gen = torch.Generator().manual_seed(21)
n = 16
base = torch.nn.functional.normalize(torch.randn(n, 32, generator=gen), dim=-1).cuda()
noise = torch.randn(n, 3, basis.shape[1], generator=gen).cuda()


def turned(base, chord, k=0):
    inside = (base @ basis) @ basis.T
    size = inside.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    d = noise[:, k] @ basis.T
    d = torch.nn.functional.normalize(d - (d * inside).sum(-1, keepdim=True) * inside / size.square(), dim=-1)
    angle = 2.0 * torch.asin((0.5 * chord / size).clamp(max=1.0))
    return torch.nn.functional.normalize((base - inside) + torch.cos(angle) * inside + torch.sin(angle) * size * d, dim=-1)


res = 200
axis = torch.linspace(-1, 1, res, device="cuda")
ga, gb = torch.meshgrid(axis, axis, indexing="xy")
plane = torch.stack([ga, gb, torch.zeros_like(ga)], -1).reshape(1, -1, 3)
show = 4
fig, axes = plt.subplots(len(models), show, figsize=(3.0 * show, 3.1 * len(models)), squeeze=False)
colors = ["#08519c", "#d95f02", "#1b9e77", "#7570b3"]
for row, (label, ckpt) in enumerate(models):
    trainer = Trainer.from_checkpoint(ckpt, device="cuda", tensorboard=False, overrides=["weak_floor=0.04", "weak_probe_codes=32"])
    cfg, model = trainer.cfg, trainer.model
    usage = trainer.measure_usage()
    lam = torch.linalg.eigvalsh(usage).flip(0).clamp_min(0).sqrt()
    inside = float((basis.T.double().cpu() @ usage @ basis.double().cpu()).diagonal().sum() / usage.diagonal().sum())
    print(f"\n== {label} (update {trainer.step_count})")
    print("  usage (RMS field change per unit chord), sorted:", " ".join(f"{v:.3f}" for v in lam.tolist()))
    print(f"  directions above 0.04: {int((lam > 0.04).sum())}; above 0.02: {int((lam > 0.02).sum())}; "
          f"share of all usage inside the directions unused at 2,500: {inside:.4f}")
    x = sample_points(cfg, 1, gen, "cuda")
    for chord in (0.1, 0.25, 0.5, 1.0):
        partner = turned(base, chord)
        fb = evaluate_field(model, torch.cat([base, partner]), torch.zeros(2 * n, dtype=torch.long, device="cuda"), x, cfg,
                            create_graph=False)
        lengths = (base - partner).norm(dim=-1)
        f, xx, g = fb.f[:, 1:], fb.x[:, 1:].detach(), fb.g[:, 1:].detach()
        mask = (xx[:n].square().sum(-1) < cfg.bound_radius ** 2).float()
        delta = (f[:n] - f[n:]).detach()
        clean = L._remove_pose(delta, xx[:n], g[:n], (0.5 * (f[:n] + f[n:])).detach(), mask)
        rate = ((clean.abs() * mask).sum(1) / mask.sum(1)) / (cfg.diversity_slope * lengths)
        near = (f[:n].detach().abs() < 0.03).float() * mask
        move = ((delta / g[:n].norm(dim=-1).clamp_min(0.3)).square() * near).sum(1) / near.sum(1).clamp_min(1)
        print(f"  chord {float(lengths.mean()):.2f}: pair rate {float(rate.mean()):.4f} (min {float(rate.min()):.4f})   "
              f"surface moves {float(move.sqrt().mean()):.4f} RMS")
        del fb
    with torch.no_grad():
        for col in range(show):
            ax = axes[row, col]
            codes = torch.cat([base[col:col + 1]] + [turned(base, 1.2, k)[col:col + 1] for k in range(3)])
            vals = model(codes, plane.expand(4, -1, -1)).reshape(4, res, res).cpu().numpy()
            a = axis.cpu().numpy()
            for i in range(4):
                ax.contour(a, a, vals[i], levels=[0], colors=colors[i], linewidths=2.0 if i == 0 else 1.1)
            ax.set_aspect("equal"); ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(label, fontsize=9)
            if row == 0:
                ax.set_title(f"code {col}", fontsize=9)
    del trainer
    torch.cuda.empty_cache()
fig.suptitle("Slice z = 0. Thick blue: a code. Thin lines: three codes with the same used part\n"
             "and the unused part (as of update 2,500) turned by chord 1.2", fontsize=10)
fig.tight_layout()
out = f"/home/eric-luo/Downloads/data_free_weak_pairs_{tag}.png"
fig.savefig(out, dpi=110)
print("saved", out)
