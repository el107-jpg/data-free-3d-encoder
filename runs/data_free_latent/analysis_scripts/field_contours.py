"""How far is the decoder's field from a true distance field? Contours and an error map on one slice.

For a few random codes: the decoder's field on the plane z = 0, and the true signed distance to the
decoder's OWN surface (the surface is extracted on a 3-D grid; distance is to its nearest point, signed
by inside/outside). Top row: contour lines of both. Bottom row: decoder field minus true distance.

usage: field_contours.py <checkpoint> <out.png> <title> [count] [seed]
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import cKDTree
from skimage import measure

from arm_planner.data_free.train import Trainer

ckpt, out_path, title = sys.argv[1], sys.argv[2], sys.argv[3]
count = int(sys.argv[4]) if len(sys.argv) > 4 else 3
seed = int(sys.argv[5]) if len(sys.argv) > 5 else 5
trainer = Trainer.from_checkpoint(ckpt, device="cuda", tensorboard=False)
cfg, model = trainer.cfg, trainer.model.eval()
gen = torch.Generator().manual_seed(seed)
codes = torch.nn.functional.normalize(torch.randn(count, cfg.latent_dim, generator=gen), dim=-1).cuda()

R3, R2 = 160, 320
ax3 = torch.linspace(-1, 1, R3, device="cuda")
vol = torch.stack(torch.meshgrid(ax3, ax3, ax3, indexing="ij"), -1).reshape(-1, 3)
ax2 = np.linspace(-1, 1, R2)
gx, gy = np.meshgrid(ax2, ax2, indexing="xy")
plane = torch.from_numpy(np.stack([gx, gy, np.zeros_like(gx)], -1).reshape(-1, 3)).float().cuda()


def field(code, pts):
    out = []
    with torch.no_grad():
        for s in range(0, pts.shape[0], 250_000):
            out.append(model(code[None], pts[None, s:s + 250_000])[0])
    return torch.cat(out).cpu().numpy()


INK, MUTED, SHAPE = "#1f1f1e", "#8a8985", "#dfe8f3"
diverge = LinearSegmentedColormap.from_list("blue_gray_red", ["#1d5fa8", "#7fa8d6", "#f0efec", "#e39a86", "#b3261e"])
levels = np.arange(-0.8, 0.45, 0.1)
fig, axes = plt.subplots(2, count, figsize=(5.0 * count, 10.4), squeeze=False)
fig.patch.set_facecolor("#fcfcfb")
summary = []
for c in range(count):
    grid = field(codes[c], vol).reshape(R3, R3, R3)
    verts, _, _, _ = measure.marching_cubes(grid, level=0.0)
    verts = verts / (R3 - 1) * 2.0 - 1.0
    f = field(codes[c], plane).reshape(R2, R2)
    dist, _ = cKDTree(verts).query(plane.cpu().numpy(), workers=-1)
    true = np.where(f > 0, 1.0, -1.0) * dist.reshape(R2, R2)
    err = f - true
    ball = (gx ** 2 + gy ** 2) < float(cfg.bound_radius) ** 2
    near = ball & (np.abs(true) < 0.1)
    summary.append((float(np.abs(err[ball]).mean()), float(np.abs(err[near]).mean()), float(np.abs(err[ball]).max())))

    ax = axes[0, c]
    ax.contourf(gx, gy, f, levels=[0, 10], colors=[SHAPE])
    cs_true = ax.contour(gx, gy, true, levels=levels, colors=MUTED, linewidths=1.0, linestyles="dashed")
    cs_dec = ax.contour(gx, gy, f, levels=levels, colors=INK, linewidths=1.1, linestyles="solid")
    ax.contour(gx, gy, f, levels=[0], colors=INK, linewidths=2.4)
    marked = [v for v in cs_dec.levels if round(float(v), 1) in (-0.6, -0.4, -0.2, 0.2)]
    ax.clabel(cs_dec, levels=marked, fmt="%.1f", fontsize=8, colors=INK)
    ax.set_title(f"random code {c + 1}: field contours every 0.1", fontsize=11, color=INK)

    ax = axes[1, c]
    im = ax.imshow(err, extent=[-1, 1, -1, 1], origin="lower", cmap=diverge, vmin=-0.1, vmax=0.1, interpolation="bilinear")
    ax.contour(gx, gy, f, levels=[0], colors=INK, linewidths=1.8)
    ax.set_title(f"decoder field minus true distance\nmean error {summary[-1][0]:.3f} (within 0.1 of the surface {summary[-1][1]:.3f})",
                 fontsize=10, color=INK)
    for a in (axes[0, c], axes[1, c]):
        a.set_aspect("equal"); a.set_xlim(-1, 1); a.set_ylim(-1, 1); a.set_xticks([-1, 0, 1]); a.set_yticks([-1, 0, 1])
        a.tick_params(colors=MUTED, labelsize=8)
        for side in a.spines.values():
            side.set_color("#d5d4cf")
handles = [plt.Line2D([], [], color=INK, lw=2.4, label="surface (field = 0)"),
           plt.Line2D([], [], color=INK, lw=1.1, label="decoder's field"),
           plt.Line2D([], [], color=MUTED, lw=1.0, ls="dashed", label="true distance to that surface")]
axes[0, 0].legend(handles=handles, loc="lower left", fontsize=8, frameon=True, facecolor="#fcfcfb", edgecolor="#d5d4cf")
bar = fig.colorbar(im, ax=axes[1, :].tolist(), orientation="horizontal", fraction=0.045, pad=0.07, aspect=50)
bar.set_label("error: red = decoder says deeper inside / nearer than it is, blue = further outside than it is (clipped at ±0.1; shape radius about 0.5)",
              fontsize=9, color=INK)
bar.ax.tick_params(colors=MUTED, labelsize=8)
fig.suptitle(f"{title} (update {trainer.step_count}): slice z = 0 through three random shapes\n"
             "Where solid and dashed lines coincide the field is a true distance", fontsize=13, color=INK, y=0.985)
fig.savefig(out_path, dpi=110, facecolor=fig.get_facecolor(), bbox_inches="tight")
print("saved", out_path)
for c, (m, n, w) in enumerate(summary):
    print(f"  code {c + 1}: mean |error| in the slice {m:.4f}, within 0.1 of the surface {n:.4f}, worst {w:.3f}")
