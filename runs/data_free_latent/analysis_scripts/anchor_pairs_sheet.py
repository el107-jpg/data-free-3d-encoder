"""Anchor objects next to what the decoder makes at their codes, for a chosen range of anchors.

usage: anchor_pairs_sheet.py <checkpoint> <out.png> <first> <last (exclusive)> <title>
Each anchor is a pair of pictures, the object (tan) above the decoded shape (blue), with the overlap printed.
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

ckpt, out_path, first, last, title = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
trainer = Trainer.from_checkpoint(ckpt, device="cuda", tensorboard=False)
cfg, model = trainer.cfg, trainer.model.eval()
codes = torch.nn.functional.normalize(trainer.anchor_codes.detach(), dim=-1)
grids = trainer.anchor_grid.float()
names = {}
meta = os.path.join(os.path.dirname(str(cfg.anchor_path)), "anchors.json")
if os.path.exists(meta):
    names = {r["rank"]: r["family"] for r in json.load(open(meta))["anchors"]}

res, extent = 96, 0.9
elev, azim = np.deg2rad(22.0), np.deg2rad(35.0)
w = np.array([np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)])
u = np.cross([0.0, 0.0, 1.0], w); u /= np.linalg.norm(u)
v = np.cross(w, u)
axis = torch.linspace(-extent, extent, res, device="cuda")
a, b_, c_ = torch.meshgrid(axis, axis, axis, indexing="ij")
frame = torch.tensor(np.stack([u, v, w]), dtype=torch.float32, device="cuda")
view = (a[..., None] * frame[0] + b_[..., None] * frame[1] + c_[..., None] * frame[2]).reshape(1, -1, 3)
cell = 2 * extent / (res - 1)
light = torch.tensor([-0.45, 0.55, 0.70], device="cuda"); light = light / light.norm()
PAPER, INK, MUTED = np.array([252, 252, 251]) / 255, "#1f1f1e", "#8a8985"
BLUE, TAN = np.array([0x7f, 0xa8, 0xd6]) / 255, np.array([0xc9, 0xa2, 0x6b]) / 255
gen = torch.Generator().manual_seed(4)
x = (torch.rand(60000, 3, generator=gen) * 2 - 1).cuda()


def render(value_fn, colour):
    with torch.no_grad():
        F = torch.cat([value_fn(view[:, s:s + 250_000]) for s in range(0, view.shape[1], 250_000)]).reshape(res, res, res)
        gu, gv, gw = torch.gradient(F, spacing=cell)
        first_hit = (F.flip(-1) > 0).float().argmax(-1)
        idx = (res - 1 - first_hit)[..., None]
        normal = -torch.stack([gu.gather(-1, idx), gv.gather(-1, idx), gw.gather(-1, idx)], -1)[..., 0, :]
        normal = normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        shade = 0.38 + 0.62 * (normal @ light).clamp_min(0.0)
        cover = (F.max(-1).values / cell + 0.5).clamp(0, 1)
        img = cover[..., None] * shade[..., None] * torch.tensor(colour, dtype=torch.float32, device="cuda") \
            + (1 - cover[..., None]) * torch.tensor(PAPER, dtype=torch.float32, device="cuda")
    return np.clip(img.permute(1, 0, 2).flip(0).cpu().numpy(), 0, 1)


picks = list(range(first, min(last, codes.shape[0])))
cols = 8
rows = int(np.ceil(len(picks) / cols))
fig, axes = plt.subplots(2 * rows, cols, figsize=(2.2 * cols, 2.45 * 2 * rows), squeeze=False)
fig.patch.set_facecolor(PAPER)
for ax in axes.ravel():
    ax.set_axis_off()
for n, k in enumerate(picks):
    r, c = divmod(n, cols)
    with torch.no_grad():
        ours = model(codes[k:k + 1], x[None])[0] > 0
        theirs = L.sample_grid(grids[k:k + 1], x[None])[0] > 0
        iou = float((ours & theirs).sum() / (ours | theirs).sum().clamp_min(1))
    axes[2 * r, c].imshow(render(lambda p, k=k: L.sample_grid(grids[k:k + 1], p)[0], TAN))
    axes[2 * r, c].set_title(f"{k}  {names.get(k, '')}", fontsize=9, color=INK, pad=2)
    axes[2 * r + 1, c].imshow(render(lambda p, k=k: model(codes[k:k + 1], p)[0], BLUE))
    axes[2 * r + 1, c].set_title(f"decoded, overlap {iou:.2f}", fontsize=8.5, color=MUTED, pad=2)
fig.suptitle(f"{title} (update {trainer.step_count}): anchor objects (tan) and what the decoder makes at their codes (blue)",
             fontsize=12.5, color=INK, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.975])
fig.savefig(out_path, dpi=100, facecolor=fig.get_facecolor())
print("saved", out_path)
