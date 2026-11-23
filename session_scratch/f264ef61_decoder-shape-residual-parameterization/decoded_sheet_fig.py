"""3-D contact sheets of what a trained model decodes.

usage: decoded_sheet.py <checkpoint> <tag>
Writes two pictures to /home/eric-luo/Downloads/pointcloud-motion-planning/figures:
  data_free_random_shapes_<tag>.png   64 fresh random codes
  data_free_anchor_fit_<tag>.png      (models with anchors) the first 16 anchor objects, each above what the
                                      decoder produces at its code
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage import measure

from arm_planner.data_free.train import Trainer

ckpt, tag = sys.argv[1], sys.argv[2]
OUT = "/home/eric-luo/Downloads/pointcloud-motion-planning/figures"
trainer = Trainer.from_checkpoint(ckpt, device="cuda", tensorboard=False)
cfg, model = trainer.cfg, trainer.model.eval()
res = 72
axis = torch.linspace(-1, 1, res, device="cuda")
pts = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(1, -1, 3)


def decode(codes):
    out = []
    with torch.no_grad():
        for c in codes:
            out.append(model(c[None], pts).reshape(res, res, res).cpu().numpy())
    return out


def draw(ax, grid, color, title=None):
    grid = np.asarray(grid, dtype=np.float32)
    if grid.max() <= 0 or grid.min() >= 0:
        ax.set_axis_off()
        return
    verts, tris, _, _ = measure.marching_cubes(grid, level=0.0)
    verts = verts / (grid.shape[0] - 1) * 2.0 - 1.0
    ax.plot_trisurf(verts[:, 0], verts[:, 1], tris, verts[:, 2], color=color, edgecolor="none", linewidth=0,
                    antialiased=False, shade=True)
    ax.set_xlim(-0.6, 0.6); ax.set_ylim(-0.6, 0.6); ax.set_zlim(-0.6, 0.6)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=22, azim=35); ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=7, pad=-4)


gen = torch.Generator().manual_seed(2024)
codes = torch.nn.functional.normalize(torch.randn(64, cfg.latent_dim, generator=gen), dim=-1).cuda()
fig = plt.figure(figsize=(2.1 * 8, 2.2 * 8))
for k, grid in enumerate(decode(codes)):
    draw(fig.add_subplot(8, 8, k + 1, projection="3d"), grid, "#7fa8d6")
fig.suptitle(f"{tag}, update {trainer.step_count}: shapes decoded from 64 fresh random codes", fontsize=12)
fig.tight_layout()
fig.savefig(f"{OUT}/data_free_random_shapes_{tag}.png", dpi=100)
plt.close(fig)
print("saved", f"{OUT}/data_free_random_shapes_{tag}.png")

if trainer.anchor_codes is not None:
    show = min(16, trainer.anchor_codes.shape[0])
    ours = decode(torch.nn.functional.normalize(trainer.anchor_codes.detach()[:show], dim=-1))
    theirs = trainer.anchor_grid[:show].float().cpu().numpy()
    fig = plt.figure(figsize=(2.1 * 8, 2.2 * 4))
    for k in range(show):
        row, col = divmod(k, 8)
        draw(fig.add_subplot(4, 8, 2 * row * 8 + col + 1, projection="3d"), theirs[k][::2, ::2, ::2] if theirs[k].shape[0] > 80 else theirs[k],
             "#c9a26b", f"object {k}")
        draw(fig.add_subplot(4, 8, (2 * row + 1) * 8 + col + 1, projection="3d"), ours[k], "#7fa8d6", "decoded")
    fig.suptitle(f"{tag}, update {trainer.step_count}: anchor objects (tan) and what the decoder makes at their codes (blue)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/data_free_anchor_fit_{tag}.png", dpi=100)
    plt.close(fig)
    print("saved", f"{OUT}/data_free_anchor_fit_{tag}.png")
