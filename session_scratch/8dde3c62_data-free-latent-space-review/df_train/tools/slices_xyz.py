"""x=0, y=0 and z=0 cross-sections of the field for 8 probe codes of a checkpoint."""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from arm_planner.data_free.train import Trainer

ckpt, out = sys.argv[1], sys.argv[2]
title = sys.argv[3] if len(sys.argv) > 3 else ""
trainer = Trainer.from_checkpoint(ckpt, device="cuda", tensorboard=False)
model = trainer.model.eval()
count = 8
codes = trainer.bank.codes[:count]
res = 256
axis = torch.linspace(-1, 1, res, device="cuda")
ga, gb = torch.meshgrid(axis, axis, indexing="xy")
zero = torch.zeros_like(ga)
planes = [("x = 0 (y across, z up)", torch.stack([zero, ga, gb], dim=-1)),
          ("y = 0 (x across, z up)", torch.stack([ga, zero, gb], dim=-1)),
          ("z = 0 (x across, y up)", torch.stack([ga, gb, zero], dim=-1))]
fig, axes = plt.subplots(3, count, figsize=(2.6 * count, 8.6))
theta = np.linspace(0, 2 * np.pi, 200)
a = axis.cpu()
for row, (name, pts) in enumerate(planes):
    with torch.no_grad():
        f = model(codes, pts.reshape(1, -1, 3).expand(count, -1, -1)).reshape(count, res, res).cpu().numpy()
    for i in range(count):
        ax = axes[row, i]
        ax.contourf(a, a, f[i], levels=[0, 10], colors=["#9ecae1"])
        ax.contour(a, a, f[i], levels=[-0.3, -0.2, -0.1, 0.1, 0.2, 0.3], colors="0.6", linewidths=0.5)
        ax.contour(a, a, f[i], levels=[0], colors="#08519c", linewidths=1.8)
        ax.plot(0.5 * np.cos(theta), 0.5 * np.sin(theta), color="#d95f02", lw=0.8, ls="--")
        ax.set_aspect("equal"); ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
        ax.set_xticks([]); ax.set_yticks([])
        if row == 0:
            ax.set_title(f"probe code {i}", fontsize=9)
        if i == 0:
            ax.set_ylabel(name, fontsize=9)
fig.suptitle(f"Three orthogonal slices, step {trainer.step_count}. Blue: shape (f > 0) and its surface; "
             f"grey: field level sets every 0.1; dashed orange: initial sphere R = 0.5\n{title}", fontsize=10)
fig.tight_layout()
fig.savefig(out, dpi=110)
print("saved", out)
