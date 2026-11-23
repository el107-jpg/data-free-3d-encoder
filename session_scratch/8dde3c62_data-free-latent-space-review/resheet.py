"""Redraw the anchor contact sheet from the saved grids (clear title, larger panels)."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage import measure

D = "/home/eric-luo/Downloads/pointcloud-motion-planning/datasets/data_free_anchors_20261111_v1/"
grids = torch.load(D + "anchors.pt", weights_only=False)["grid"].float().numpy()
meta = json.load(open(D + "anchors.json"))["anchors"]
count, cols = grids.shape[0], 8
rows = int(np.ceil(count / cols))
fig = plt.figure(figsize=(2.1 * cols, 2.25 * rows + 0.6))
res = grids.shape[1]
for k in range(count):
    ax = fig.add_subplot(rows, cols, k + 1, projection="3d")
    verts, tris, _, _ = measure.marching_cubes(grids[k], level=0.0, step_size=2)
    verts = verts / (res - 1) * 2.0 - 1.0
    ax.plot_trisurf(verts[:, 0], verts[:, 1], tris, verts[:, 2], color="#7fa8d6", edgecolor="none", linewidth=0,
                    antialiased=False, shade=True)
    ax.set_xlim(-0.55, 0.55); ax.set_ylim(-0.55, 0.55); ax.set_zlim(-0.55, 0.55)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=22, azim=35); ax.set_axis_off()
    name = str(meta[k]["name"]).replace("&#039;", "'")[:24]
    ax.set_title(f"{k}: {name}", fontsize=7, pad=-6)
fig.suptitle("64 anchor objects from Thingi10K, in selection order (each is the one least like those before it)",
             fontsize=12, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.975])
fig.savefig("/home/eric-luo/Downloads/data_free_anchor_objects.png", dpi=110)
print("saved")
