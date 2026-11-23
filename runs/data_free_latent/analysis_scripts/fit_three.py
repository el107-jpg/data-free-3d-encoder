"""Fit a code by gradient descent to a few random Thingi objects the model never trained on; one picture.

usage: fit_three.py <checkpoint> <out.png> [count] [seed]
Targets are drawn at random from the Thingi objects that are valid single solids (the pool built without the
filled-centre filter), leaving out this model's own anchor objects.
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage import measure

from arm_planner.data_free import losses as L
from arm_planner.data_free.train import Trainer

ckpt, out_path = sys.argv[1], sys.argv[2]
count = int(sys.argv[3]) if len(sys.argv) > 3 else 3
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
D = "/home/eric-luo/Downloads/pointcloud-motion-planning/datasets/"
pool = torch.load(D + "data_free_anchors_20261112_v3/eligible.pt", map_location="cpu", weights_only=False)
own = torch.load(D + "data_free_anchors_20261111_v1/anchors.pt", map_location="cpu", weights_only=False)
own_ids = set(int(i) for i in own["file_ids"])
free = [k for k, i in enumerate(pool["file_ids"]) if int(i) not in own_ids]
gen = torch.Generator().manual_seed(seed)
pick = [free[int(j)] for j in torch.randperm(len(free), generator=gen)[:count]]
names = [str(pool["names"][k]).replace("&#039;", "'")[:28] for k in pick] if "names" in pool else [str(k) for k in pick]

trainer = Trainer.from_checkpoint(ckpt, device="cuda", tensorboard=False)
cfg, model = trainer.cfg, trainer.model.eval()
for p in model.parameters():
    p.requires_grad_(False)
if getattr(cfg, "anchor_path", ""):
    # never score the model on one of its own anchor objects
    mine = set(int(i) for i in torch.load(str(cfg.anchor_path), map_location="cpu", weights_only=False)["file_ids"])
    seen = [k for k in pick if int(pool["file_ids"][k]) in mine]
    if seen:
        print("left out (this model's own anchors):", [names[pick.index(k)] for k in seen], flush=True)
        names = [n for n, k in zip(names, pick) if k not in seen]
        pick = [k for k in pick if k not in seen]
        count = len(pick)
dim, bound = int(cfg.latent_dim), float(cfg.bound_radius)
starts, steps, points = 8, 400, 4096
test_x = (torch.rand(30000, 3, generator=gen) * 2 - 1).cuda()
res = 72
axis = torch.linspace(-1, 1, res, device="cuda")
grid_pts = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(1, -1, 3)

fits, scores = [], []
for k in pick:
    target = pool["grid"][k:k + 1].float().cuda()
    codes = torch.nn.functional.normalize(torch.randn(starts, dim, generator=gen), dim=-1).cuda().requires_grad_(True)
    opt = torch.optim.Adam([codes], lr=0.03)
    for step in range(steps):
        x = (torch.rand(1, points, 3, generator=gen) * 2 - 1).cuda().expand(starts, -1, -1)
        inside = (x.square().sum(-1) < bound ** 2).float()
        f = model(torch.nn.functional.normalize(codes, dim=-1), x)
        t = L.sample_grid(target.expand(starts, -1, -1, -1), x)
        loss = (((f - t).abs() * inside).sum(1) / inside.sum(1)).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        for group in opt.param_groups:
            group["lr"] = 0.03 * (0.1 ** (step / steps))
    with torch.no_grad():
        unit = torch.nn.functional.normalize(codes, dim=-1)
        xt = test_x[None].expand(starts, -1, -1)
        f = model(unit, xt)
        t = L.sample_grid(target.expand(starts, -1, -1, -1), xt)
        ins = xt.square().sum(-1) < bound ** 2
        err = ((f - t).abs() * ins).sum(1) / ins.sum(1)
        best = int(err.argmin())
        ours, theirs = f[best] > 0, t[best] > 0
        iou = float((ours & theirs).sum() / (ours | theirs).sum().clamp_min(1))
        scores.append(iou)
        fits.append(model(unit[best:best + 1], grid_pts).reshape(res, res, res).cpu().numpy())
    print(f"{names[len(scores) - 1]}: overlap {iou:.2f}, field error {float(err[best]):.4f}", flush=True)


def draw(ax, grid, color, title):
    grid = np.asarray(grid, dtype=np.float32)
    verts, tris, _, _ = measure.marching_cubes(grid, level=0.0)
    verts = verts / (grid.shape[0] - 1) * 2.0 - 1.0
    ax.plot_trisurf(verts[:, 0], verts[:, 1], tris, verts[:, 2], color=color, edgecolor="none", linewidth=0,
                    antialiased=False, shade=True)
    ax.set_xlim(-0.6, 0.6); ax.set_ylim(-0.6, 0.6); ax.set_zlim(-0.6, 0.6)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=22, azim=35); ax.set_axis_off()
    ax.set_title(title, fontsize=10, pad=-2)


fig = plt.figure(figsize=(4.2 * count, 8.6))
for c, k in enumerate(pick):
    draw(fig.add_subplot(2, count, c + 1, projection="3d"), pool["grid"][k].float().numpy()[::2, ::2, ::2], "#c9a26b",
         f"target: {names[c]}")
    draw(fig.add_subplot(2, count, count + c + 1, projection="3d"), fits[c], "#7fa8d6", f"decoder's best fit, overlap {scores[c]:.2f}")
fig.suptitle(f"Random Thingi objects the model never trained on (top) and the closest shape found by\n"
             f"gradient descent on the code (bottom); model at update {trainer.step_count}", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(out_path, dpi=110)
print("saved", out_path)
