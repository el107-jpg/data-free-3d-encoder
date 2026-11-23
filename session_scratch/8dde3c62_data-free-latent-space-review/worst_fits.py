"""Which unseen objects does a model imitate worst, and what do they have in common?

usage: worst_fits.py <checkpoint> <fit tag> <out.png> [how many to draw]
Reads <fit tag>.json and .pt written by heldout_fit.py (best code per unseen object).

 1. Draws the worst-fitting objects (tan) beside the decoder's best attempt (blue).
 2. Measures every unseen object's shape descriptors (scripts/data_free_shape_descriptors.py) and reports
    how each one goes with the overlap: rank correlation, and the mean overlap of the lowest and highest third.
"""
import json
import multiprocessing as mp
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, "scripts")
import data_free_shape_descriptors as SD  # noqa: E402

from arm_planner.data_free import losses as L  # noqa: E402
from arm_planner.data_free.train import Trainer  # noqa: E402

ckpt, tag, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
show = int(sys.argv[4]) if len(sys.argv) > 4 else 16
D = "/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/analysis_scripts/followup/"
pool = torch.load("/home/eric-luo/Downloads/pointcloud-motion-planning/datasets/data_free_anchors_20261112_v3/eligible.pt",
                  map_location="cpu", weights_only=False)
fit = json.load(open(D + tag + ".json"))["held_out"]
saved = torch.load(D + tag + ".pt", map_location="cpu", weights_only=False)
index, codes = saved["held_index"], saved["held_codes"]
iou = np.array(fit["per_object_iou"])
names = [str(pool["names"][k]).replace("&#039;", "'").replace("&amp;", "&")[:26] for k in index]
grids = pool["grid"]


def describe(k):
    return SD.describe(grids[k].float().numpy(), seed=int(k))


if __name__ == "__main__":
    with mp.get_context("fork").Pool(10) as workers:
        desc = workers.map(describe, index, chunksize=4)
    volume = np.array([float((grids[k].float() > 0).float().mean()) * 8.0 for k in index])
    rank = lambda v: np.argsort(np.argsort(v))
    print(f"{tag}: {len(iou)} unseen objects, mean overlap {iou.mean():.3f}\n")
    print(f"{'property of the object':46s} {'rank correlation':>17s} {'lowest third':>13s} {'middle':>8s} {'highest third':>14s}")
    rows = [("volume", volume)] + [(key, np.array([d[key] for d in desc])) for key in SD.KEYS]
    label = {"volume": "volume", "linear": "rod-like share of the surface", "planar": "sheet-like share", "round": "blob-like share",
             "thickness": "thickness (typical width across)", "ray_crossings": "surface crossings per ray from the centre",
             "star_fraction": "share of rays crossing once (star-shaped)", "fold": "fold (surface area against its convex hull's)",
             "solidity": "solidity (volume against its convex hull's)", "branches": "number of branches", "gap": "narrowest gap (0 = none)"}
    for key, v in rows:
        order = np.argsort(v)
        third = len(v) // 3
        corr = float(np.corrcoef(rank(v), rank(iou))[0, 1])
        print(f"{label.get(key, key):46s} {corr:+17.2f} {iou[order[:third]].mean():13.3f} {iou[order[third:2 * third]].mean():8.3f} {iou[order[2 * third:]].mean():14.3f}")
    worst = np.argsort(iou)[:show]
    print("\nthe worst-fitting objects:")
    for n in worst:
        d = desc[n]
        print(f"  {iou[n]:.2f}  {names[n]:28s} volume {volume[n]:.3f}  thickness {d['thickness']:.2f}  rod {d['linear']:.2f} sheet {d['planar']:.2f} blob {d['round']:.2f}  "
              f"ray crossings {d['ray_crossings']:.1f}  solidity {d['solidity']:.2f}  branches {d['branches']:.0f}")

    trainer = Trainer.from_checkpoint(ckpt, device="cuda", tensorboard=False)
    model = trainer.model.eval()
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

    def render(value_fn, colour):
        with torch.no_grad():
            F = torch.cat([value_fn(view[:, s:s + 250_000]) for s in range(0, view.shape[1], 250_000)]).reshape(res, res, res)
            gu, gv, gw = torch.gradient(F, spacing=cell)
            hit = (F.flip(-1) > 0).float().argmax(-1)
            idx = (res - 1 - hit)[..., None]
            normal = -torch.stack([gu.gather(-1, idx), gv.gather(-1, idx), gw.gather(-1, idx)], -1)[..., 0, :]
            normal = normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            shade = 0.38 + 0.62 * (normal @ light).clamp_min(0.0)
            cover = (F.max(-1).values / cell + 0.5).clamp(0, 1)
            img = cover[..., None] * shade[..., None] * torch.tensor(colour, dtype=torch.float32, device="cuda") \
                + (1 - cover[..., None]) * torch.tensor(PAPER, dtype=torch.float32, device="cuda")
        return np.clip(img.permute(1, 0, 2).flip(0).cpu().numpy(), 0, 1)

    cols = 8
    rows_n = int(np.ceil(show / cols))
    fig, axes = plt.subplots(2 * rows_n, cols, figsize=(2.2 * cols, 2.45 * 2 * rows_n), squeeze=False)
    fig.patch.set_facecolor(PAPER)
    for ax in axes.ravel():
        ax.set_axis_off()
    for n, j in enumerate(worst):
        r, c = divmod(n, cols)
        target = grids[index[j]:index[j] + 1].float().cuda()
        code = codes[j:j + 1].cuda()
        axes[2 * r, c].imshow(render(lambda p: L.sample_grid(target, p)[0], TAN))
        axes[2 * r, c].set_title(names[j], fontsize=8.5, color=INK, pad=2)
        axes[2 * r + 1, c].imshow(render(lambda p: model(code, p)[0], BLUE))
        axes[2 * r + 1, c].set_title(f"best attempt, overlap {iou[j]:.2f}", fontsize=8.5, color=MUTED, pad=2)
    fig.suptitle(f"The {show} unseen Thingi objects this model imitates worst (tan), each above the closest shape found by "
                 f"searching for a code (blue); model at update {trainer.step_count}", fontsize=12, color=INK, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(out_path, dpi=100, facecolor=fig.get_facecolor())
    print("\nsaved", out_path)
