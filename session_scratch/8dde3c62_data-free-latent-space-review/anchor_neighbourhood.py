"""How far from an anchor's code does the decoded shape still look like the anchor?

usage: anchor_neighbourhood.py <checkpoint> <out.png> <label>

For every anchor and 4 random directions, codes at set angles from the anchor's code are decoded and compared
with the anchor object (volume overlap on 20,000 points). Baseline: the overlap between each anchor object and
shapes at unrelated random codes. Also: how far random codes sit from their nearest anchor code.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from arm_planner.data_free import losses as L
from arm_planner.data_free.train import Trainer

ckpt, out_path, label = sys.argv[1], sys.argv[2], sys.argv[3]
trainer = Trainer.from_checkpoint(ckpt, device="cuda", tensorboard=False)
cfg, model = trainer.cfg, trainer.model.eval()
dim = int(cfg.latent_dim)
anchors = torch.nn.functional.normalize(trainer.anchor_codes.detach(), dim=-1)
grids = trainer.anchor_grid.float()
count = anchors.shape[0]
gen = torch.Generator().manual_seed(12)
x = (torch.rand(20000, 3, generator=gen) * 2 - 1).cuda()
ANGLES = [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 90]
DIRS = 4


def solids(codes):
    with torch.no_grad():
        return torch.cat([model(codes[s:s + 16], x[None].expand(codes[s:s + 16].shape[0], -1, -1)) > 0 for s in range(0, codes.shape[0], 16)])


def iou(a, b):
    return ((a & b).sum(-1).float() / (a | b).sum(-1).clamp_min(1).float())


with torch.no_grad():
    target = torch.cat([L.sample_grid(grids[k:k + 8], x[None].expand(grids[k:k + 8].shape[0], -1, -1)) > 0 for k in range(0, count, 8)])   # K x M
curve = np.zeros((count, len(ANGLES)))
for k in range(count):
    a = anchors[k]
    d = torch.randn(DIRS, dim, generator=gen).cuda()
    d = torch.nn.functional.normalize(d - (d @ a)[:, None] * a, dim=-1)
    for j, deg in enumerate(ANGLES):
        t = np.deg2rad(deg)
        codes = float(np.cos(t)) * a[None] + float(np.sin(t)) * d
        curve[k, j] = float(iou(solids(codes), target[k][None]).mean())
random_codes = torch.nn.functional.normalize(torch.randn(128, dim, generator=gen), dim=-1).cuda()
rand_solid = solids(random_codes)
base = np.array([float(iou(rand_solid, target[k][None]).mean()) for k in range(count)])

groups = [("all anchors", slice(0, count))]
if count == 96:
    groups = [("64 thick anchors", slice(0, 64)), ("16 thin rods", slice(64, 80)), ("16 thin sheets", slice(80, 96))]
print(f"{label} (update {trainer.step_count}); angle from the anchor's code -> overlap with the anchor object")
print(f"{'':20s}" + "".join(f"{a:>6d}" for a in ANGLES) + "   unrelated random codes   half-way angle")
half = {}
for name, s in groups + ([("all 96", slice(0, count))] if count == 96 else []):
    m, b = curve[s].mean(0), base[s].mean()
    mid = 0.5 * (m[0] + b)
    j = int(np.argmax(m < mid))
    angle = ANGLES[j - 1] + (m[j - 1] - mid) / (m[j - 1] - m[j]) * (ANGLES[j] - ANGLES[j - 1]) if j > 0 else 0.0
    half[name] = angle
    print(f"{name:20s}" + "".join(f"{v:6.2f}" for v in m) + f"   {b:10.2f}              {angle:6.1f} deg (cosine {np.cos(np.deg2rad(angle)):.2f})")

# how far random codes are from their nearest anchor
many = torch.nn.functional.normalize(torch.randn(200000, dim, generator=gen), dim=-1).cuda()
nearest = (many @ anchors.T).max(dim=1).values.clamp(-1, 1)
ang = torch.rad2deg(torch.arccos(nearest)).cpu().numpy()
print(f"\nrandom codes: angle to the nearest anchor code: median {np.median(ang):.1f} deg, 1st percentile {np.percentile(ang, 1):.1f}, smallest of 200,000: {ang.min():.1f}")
for deg in (20, 30, 40, 50, 60):
    print(f"  share of random codes within {deg} deg of an anchor: {(ang < deg).mean():.2e}")
radius = float(getattr(cfg, "anchor_radius", 0.0))
pull_deg = float(np.rad2deg(2 * np.arcsin(min(1.0, radius / 2)))) if radius > 0 else 0.0
far_deg = float(np.rad2deg(2 * np.arcsin(min(1.0, radius)))) if radius > 0 else 0.0
print(f"training: codes around each anchor are sampled out to {far_deg:.0f} deg; the pull is down to 37% at {pull_deg:.0f} deg and 2% at {far_deg:.0f} deg")

INK, MUTED, GRIDC, PAPER = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a"]
fig, (ax, bx) = plt.subplots(1, 2, figsize=(14.2, 5.5), gridspec_kw={"width_ratios": [1.5, 1]})
fig.patch.set_facecolor(PAPER)
for (name, s), colour in zip(groups, COLOURS):
    m, b = curve[s].mean(0), base[s].mean()
    ax.plot(ANGLES, m, color=colour, lw=2, marker="o", ms=4, label=name)
    ax.plot([ANGLES[0], ANGLES[-1]], [b, b], color=colour, lw=1, ls=":")
    ax.text(ANGLES[-1] + 1, m[-1], f"{m[-1]:.2f}", fontsize=8.5, color=INK, va="center")
ax.plot([], [], color=MUTED, lw=1, ls=":", label="same anchors against unrelated random codes")
if pull_deg > 0:
    ax.axvline(pull_deg, color=MUTED, lw=1, ls="--")
    ax.text(pull_deg + 0.8, 0.03, f"training pull down to 37% ({pull_deg:.0f}°)", fontsize=8.5, color=MUTED, rotation=90, va="bottom")
ax.set_xlim(0, 96); ax.set_ylim(0, 1); ax.set_xticks([0, 15, 30, 45, 60, 75, 90])
ax.set_xlabel("angle between the code and the anchor's code (degrees)", fontsize=9.5, color=MUTED)
ax.set_ylabel("overlap of the decoded shape with the anchor object", fontsize=9.5, color=MUTED)
ax.set_title(f"{label}: how quickly the shape stops looking like the anchor\nas the code moves away from the anchor's code", fontsize=11, color=INK, loc="left")
ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper right")
bx.hist(ang, bins=np.arange(40, 91, 1), color="#2a78d6", edgecolor=PAPER, linewidth=0.6)
bx.set_xlim(0, 96); bx.set_xticks([0, 15, 30, 45, 60, 75, 90]); bx.set_yticks([])
bx.set_xlabel("angle from a random code to the nearest anchor code (degrees)", fontsize=9.5, color=MUTED)
bx.set_title(f"Where random codes sit: median {np.median(ang):.0f}° from the nearest anchor,\nnone of 200,000 closer than {ang.min():.0f}°", fontsize=11, color=INK, loc="left")
for cx in (ax, bx):
    cx.set_facecolor(PAPER); cx.grid(True, color=GRIDC, lw=0.6); cx.set_axisbelow(True); cx.tick_params(colors=MUTED, labelsize=8.5, length=0)
    for side in cx.spines.values():
        side.set_visible(False)
fig.tight_layout()
fig.savefig(out_path, dpi=115, facecolor=fig.get_facecolor())
print("saved", out_path)
