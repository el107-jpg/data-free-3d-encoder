"""Held-out targets beside the best fit each model found (eight objects spread from best to worst fit).

usage: fit_picture.py <eligible.pt> <out.png> <label>=<tag>=<checkpoint> ...
The first model's overlap decides which objects are shown.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage import measure

from arm_planner.data_free.train import Trainer

HERE = os.path.dirname(os.path.abspath(__file__))
elig = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
out_path = sys.argv[2]
models = [m.split("=") for m in sys.argv[3:]]
first = json.load(open(os.path.join(HERE, models[0][1] + ".json")))
ious = np.array(first["held_out"]["per_object_iou"])
order = np.argsort(-ious)
show = [int(order[int(round(q * (len(order) - 1)))]) for q in np.linspace(0, 1, 8)]
res = 72
axis = torch.linspace(-1, 1, res, device="cuda")
pts = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(1, -1, 3)


def draw(ax, grid, color, title):
    grid = np.asarray(grid, dtype=np.float32)
    if grid.max() > 0 > grid.min():
        verts, tris, _, _ = measure.marching_cubes(grid, level=0.0)
        verts = verts / (grid.shape[0] - 1) * 2.0 - 1.0
        ax.plot_trisurf(verts[:, 0], verts[:, 1], tris, verts[:, 2], color=color, edgecolor="none", linewidth=0,
                        antialiased=False, shade=True)
    ax.set_xlim(-0.6, 0.6); ax.set_ylim(-0.6, 0.6); ax.set_zlim(-0.6, 0.6)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=22, azim=35); ax.set_axis_off()
    ax.set_title(title, fontsize=8, pad=-4)


fig = plt.figure(figsize=(2.1 * 8, 2.25 * (1 + len(models)) + 0.5))
held_index = None
for row, (label, tag, ckpt) in enumerate(models):
    res_json = json.load(open(os.path.join(HERE, tag + ".json")))
    saved = torch.load(os.path.join(HERE, tag + ".pt"), map_location="cpu", weights_only=False)
    held_index = saved["held_index"]
    trainer = Trainer.from_checkpoint(ckpt, device="cuda", tensorboard=False)
    model = trainer.model.eval()
    for col, k in enumerate(show):
        if row == 0:
            target = elig["grid"][held_index[k]].float().numpy()[::2, ::2, ::2]
            draw(fig.add_subplot(1 + len(models), 8, col + 1, projection="3d"), target, "#c9a26b", "held-out object")
        with torch.no_grad():
            grid = model(saved["held_codes"][k:k + 1].cuda(), pts).reshape(res, res, res).cpu().numpy()
        draw(fig.add_subplot(1 + len(models), 8, (row + 1) * 8 + col + 1, projection="3d"), grid, "#7fa8d6",
             f'{label}: overlap {res_json["held_out"]["per_object_iou"][k]:.2f}')
    del trainer
    torch.cuda.empty_cache()
fig.suptitle("Objects the models never saw (tan) and the closest shape each model can produce (blue), "
             "from its best to its worst fit", fontsize=12, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(out_path, dpi=100)
print("saved", out_path)
