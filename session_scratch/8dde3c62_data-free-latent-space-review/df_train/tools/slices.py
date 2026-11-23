"""z=0 cross-sections of the field for the probe codes of a checkpoint."""
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
codes = trainer.bank.codes[:16]
res = 256
axis = torch.linspace(-1, 1, res, device="cuda")
gx, gy = torch.meshgrid(axis, axis, indexing="xy")
pts = torch.stack([gx, gy, torch.zeros_like(gx)], dim=-1).reshape(1, -1, 3).expand(16, -1, -1)
with torch.no_grad():
    f = model(codes, pts).reshape(16, res, res).cpu().numpy()
fig, axes = plt.subplots(4, 4, figsize=(12, 12.6))
theta = np.linspace(0, 2 * np.pi, 200)
for i, ax in enumerate(axes.flat):
    ax.contourf(axis.cpu(), axis.cpu(), f[i], levels=[0, 10], colors=["#9ecae1"])
    ax.contour(axis.cpu(), axis.cpu(), f[i], levels=[-0.3, -0.2, -0.1, 0.1, 0.2, 0.3],
               colors="0.6", linewidths=0.5)
    ax.contour(axis.cpu(), axis.cpu(), f[i], levels=[0], colors="#08519c", linewidths=1.8)
    ax.plot(0.5 * np.cos(theta), 0.5 * np.sin(theta), color="#d95f02", lw=0.8, ls="--")
    ax.set_aspect("equal"); ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"probe code {i}", fontsize=9)
fig.suptitle(f"z = 0 slice, step {trainer.step_count}. Blue: shape (f > 0) and its surface; "
             f"grey: field level sets every 0.1; dashed orange: initial sphere R = 0.5\n{title}",
             fontsize=10)
fig.tight_layout()
fig.savefig(out, dpi=110)
print("saved", out)
