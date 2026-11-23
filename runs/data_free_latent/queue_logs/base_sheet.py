"""Bases (tan) and decoded shapes (blue) of a rounded-box-base checkpoint.
usage: base_sheet.py <checkpoint> <tag>"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage import measure
from arm_planner.data_free.train import Trainer
from arm_planner.data_free.decoder import rounded_box_field

ckpt, tag = sys.argv[1], sys.argv[2]
OUT = "/home/eric-luo/Downloads/pointcloud-motion-planning/figures"
trainer = Trainer.from_checkpoint(ckpt, device="cuda", tensorboard=False)
cfg, model = trainer.cfg, trainer.model.eval()
res = 72
axis = torch.linspace(-1, 1, res, device="cuda")
pts = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(1, -1, 3)

gen = torch.Generator().manual_seed(2024)
pool = torch.nn.functional.normalize(torch.randn(1024, cfg.latent_dim, generator=gen), dim=-1).cuda()
with torch.no_grad():
    base = model.prepare(pool).base
dev = torch.cat([base[:, :3], base[:, 3:] - cfg.base_radius], -1).square().sum(-1)
spread = base[:, :3].max(-1).values - base[:, 3]           # extent relative to rounding: how box-like
pick = torch.cat([torch.arange(8, device="cuda"), spread.argsort(descending=True)[:8]])
codes, bases = pool[pick], base[pick]

def draw(ax, grid, color, title):
    if grid.max() <= 0 or grid.min() >= 0:
        ax.set_axis_off(); return
    v, t, _, _ = measure.marching_cubes(grid, level=0.0)
    v = v / (res - 1) * 2.0 - 1.0
    ax.plot_trisurf(v[:, 0], v[:, 1], t, v[:, 2], color=color, edgecolor="none", linewidth=0, antialiased=False, shade=True)
    ax.set_xlim(-0.6, 0.6); ax.set_ylim(-0.6, 0.6); ax.set_zlim(-0.6, 0.6)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=22, azim=35); ax.set_axis_off()
    ax.set_title(title, fontsize=7.5, pad=-2)

fig = plt.figure(figsize=(2.1 * 8, 2.25 * 4))
with torch.no_grad():
    for k in range(16):
        b = bases[k:k + 1]
        base_grid = model.base_field(model.prepare(codes[k:k + 1]), pts).reshape(res, res, res).cpu().numpy()
        full = model(codes[k:k + 1], pts).reshape(res, res, res).cpu().numpy()
        row, col = divmod(k, 8)
        e = b[0].tolist()
        label = "random" if k < 8 else "most box-like"
        d = model.prepare(codes[k:k + 1]).deform
        if d is not None:
            e = e + [float(d[0, :6].abs().max()), float(d[0, 6:9].abs().max())]
        draw(fig.add_subplot(4, 8, 2 * row * 8 + col + 1, projection="3d"), base_grid, "#c9a26b",
             f"{label}: base\nb=({e[0]:.2f},{e[1]:.2f},{e[2]:.2f}) r={e[3]:.2f}" + (f"\nbend {e[4]:.2f} taper {e[5]:.2f}" if len(e) > 4 else ""))
        draw(fig.add_subplot(4, 8, (2 * row + 1) * 8 + col + 1, projection="3d"), full, "#7fa8d6", "decoded shape")
fig.suptitle(f"{tag}, update {trainer.step_count}: rounded-box base (tan) and the decoded shape (blue). "
             f"Top 8: random codes; bottom 8: the most box-like of 1,024 random codes", fontsize=11)
fig.tight_layout()
path = f"{OUT}/data_free_box_base_{tag}.png"
fig.savefig(path, dpi=100)
print("saved", path)
q = lambda x: [round(float(v), 3) for v in torch.quantile(x, torch.tensor([0.05, 0.5, 0.95], device=x.device))]
print("over 1,024 codes: r", q(base[:, 3]), " largest extent", q(base[:, :3].max(-1).values), " smallest extent", q(base[:, :3].min(-1).values))
