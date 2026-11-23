"""The same random codes decoded by several models, one model per row, shaded so facets and edges show.

usage: crease_pairs.py <out.png> <title> <label>=<checkpoint> [<label>=<checkpoint> ...] [codes=8] [seed=31]
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

from arm_planner.data_free.train import Trainer

out_path, title = sys.argv[1], sys.argv[2]
models = [a.split("=", 1) for a in sys.argv[3:] if not a.startswith(("codes=", "seed="))]
opts = dict(a.split("=", 1) for a in sys.argv[3:] if a.startswith(("codes=", "seed=")))
COUNT, SEED, RES = int(opts.get("codes", 8)), int(opts.get("seed", 31)), 128
INK, MUTED, PAPER = "#0b0b0b", "#52514e", "#fcfcfb"
BASE = np.array([0.42, 0.60, 0.84])
LIGHT = np.array([0.45, -0.55, 0.70]) / np.linalg.norm([0.45, -0.55, 0.70])

axis = torch.linspace(-1, 1, RES, device="cuda")
pts = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(1, -1, 3)
fig = plt.figure(figsize=(2.15 * COUNT + 1.3, 2.3 * len(models) + 0.7))
fig.patch.set_facecolor(PAPER)
codes = None
for row, (label, path) in enumerate(models):
    trainer = Trainer.from_checkpoint(path, device="cuda", tensorboard=False)
    model = trainer.model.eval()
    if codes is None:
        gen = torch.Generator().manual_seed(SEED)
        codes = torch.nn.functional.normalize(torch.randn(COUNT, int(trainer.cfg.latent_dim), generator=gen), dim=-1).cuda()
    for col in range(COUNT):
        with torch.no_grad():
            grid = torch.cat([model(codes[col][None], pts[:, s:s + 262144]) for s in range(0, pts.shape[1], 262144)], dim=1)
        grid = grid.reshape(RES, RES, RES).cpu().numpy()
        ax = fig.add_subplot(len(models), COUNT, row * COUNT + col + 1, projection="3d")
        ax.set_facecolor(PAPER)
        ax.set_axis_off()
        if grid.max() > 0 and grid.min() < 0:
            verts, faces, _, _ = measure.marching_cubes(grid, 0.0, spacing=(2 / (RES - 1),) * 3)
            verts = verts - 1.0
            tri = verts[faces]
            normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            normal /= np.linalg.norm(normal, axis=1, keepdims=True).clip(1e-12)
            if (normal * (tri.mean(1) - verts.mean(0))).sum() < 0:      # make the normals point outward
                normal = -normal
            shade = 0.22 + 0.78 * np.clip(normal @ LIGHT, 0, 1)
            mesh = Poly3DCollection(tri, facecolors=np.clip(BASE[None] * shade[:, None] * 1.08, 0, 1), edgecolors="none", antialiased=False)
            ax.add_collection3d(mesh)
            half = max(np.abs(verts).max() * 1.02, 0.3)
            ax.set_xlim(-half, half); ax.set_ylim(-half, half); ax.set_zlim(-half, half)
        ax.set_box_aspect((1, 1, 1), zoom=1.45)
        ax.view_init(elev=22, azim=-58)
        if col == 0:
            ax.text2D(-0.02, 0.5, label, transform=ax.transAxes, rotation=90, va="center", ha="right", fontsize=10, color=INK)
        if row == 0:
            ax.set_title(f"code {col + 1}", fontsize=9, color=MUTED, pad=0)
    del trainer, model
fig.suptitle(title, fontsize=12, color=INK, x=0.02, ha="left")
fig.subplots_adjust(left=0.05, right=0.995, top=0.90, bottom=0.01, wspace=0.0, hspace=0.02)
fig.savefig(out_path, dpi=110, facecolor=fig.get_facecolor())
print("saved", out_path)
