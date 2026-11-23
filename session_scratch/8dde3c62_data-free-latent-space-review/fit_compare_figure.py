"""Pictures comparing models on the unseen-object fit (heldout_fit.py + rescore_fits.py results).

usage: fit_compare_figure.py <chart.png> <shapes.png> <label>|<fit tag>|<checkpoint> [...]
The first model is the reference for the paired differences.

chart.png: per model, mean overlap and mean surface distance over the 225 unseen objects, for all of them and for
each third by thickness (95% intervals), and the per-object difference in surface distance to the reference
model (paired, so the interval is much narrower).
shapes.png: 8 unseen objects spread over the thickness range (fixed choice: the objects at the 6th, 19th, ...,
94th percentile of thickness), each next to every model's best attempt, all drawn the same way, with the
attempt's surface distance.
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

from arm_planner.data_free.train import Trainer

D = "/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/analysis_scripts/followup/"
POOL = "/home/eric-luo/Downloads/pointcloud-motion-planning/datasets/data_free_anchors_20261112_v3/eligible.pt"
chart_path, shapes_path = sys.argv[1], sys.argv[2]
models = [a.split("|") for a in sys.argv[3:]]
INK, MUTED, GRID, PAPER = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COLOURS = ["#9a9893", "#5c5a56", "#e39a86", "#eb6834", "#2a78d6", "#1baf7a", "#8a5cc7"]
BASE = np.array([0.42, 0.60, 0.84])
LIGHT = np.array([0.45, -0.55, 0.70]) / np.linalg.norm([0.45, -0.55, 0.70])

pool = torch.load(POOL, map_location="cpu", weights_only=False)
desc = json.load(open(D + "heldout_descriptors_v3.json"))
results = []
for label, tag, ckpt in models:
    fit = torch.load(D + tag + ".pt", map_location="cpu", weights_only=False)
    surf = json.load(open(D + tag + "_surface.json"))
    index = list(fit["held_index"])
    assert [int(pool["file_ids"][k]) for k in index] == [int(v) for v in surf["file_ids"]], tag
    results.append({"label": label, "ckpt": ckpt, "codes": fit["held_codes"], "index": index,
                    "iou": np.array(surf["iou"]), "dist": np.array(surf["distance"])})
index = results[0]["index"]
for r in results:
    assert r["index"] == index
thick = np.array([desc[str(k)]["thickness"] for k in index])
order = np.argsort(thick)
thirds = [order[:75], order[75:150], order[150:]]

# ----------------------------------------------------------------- chart
fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4), gridspec_kw={"width_ratios": [1, 1, 0.8]})
fig.patch.set_facecolor(PAPER)
groups = [("all 225", np.arange(len(index))), ("thinnest third", thirds[0]), ("middle third", thirds[1]), ("thickest third", thirds[2])]
xs = np.arange(len(groups))
width = 0.8 / len(results)
for ax, key, title, better in ((axes[0], "iou", "Overlap with the object (higher is better)", "higher"),
                               (axes[1], "dist", "Surface distance to the object (lower is better)", "lower")):
    for i, (r, colour) in enumerate(zip(results, COLOURS)):
        means = [r[key][g].mean() for _, g in groups]
        ci = [1.96 * r[key][g].std(ddof=1) / np.sqrt(len(g)) for _, g in groups]
        pos = xs - 0.4 + width * (i + 0.5)
        ax.errorbar(pos, means, yerr=ci, fmt="o", color=colour, ms=6, lw=1.4, capsize=0, label=r["label"])
    ax.set_xticks(xs)
    ax.set_xticklabels([g for g, _ in groups], fontsize=9, color=MUTED)
    ax.set_title(title, fontsize=11, color=INK, loc="left")
for i, (r, colour) in enumerate(zip(results[1:], COLOURS[1:])):
    diff = r["dist"] - results[0]["dist"]
    axes[2].errorbar([diff.mean()], [i], xerr=[1.96 * diff.std(ddof=1) / np.sqrt(len(diff))], fmt="o", color=colour, ms=6, lw=1.6, capsize=0)
    axes[2].text(diff.mean(), i + 0.22, f"{diff.mean():+.4f}", fontsize=8.5, color=INK, ha="center")
axes[2].axvline(0, color=MUTED, lw=1)
axes[2].set_yticks(range(len(results) - 1))
axes[2].set_yticklabels([r["label"] for r in results[1:]], fontsize=9, color=INK)
axes[2].invert_yaxis()
axes[2].set_title(f"Surface distance minus {results[0]['label']}'s,\nsame objects (right of 0 = worse)", fontsize=11, color=INK, loc="left")
for ax in axes:
    ax.set_facecolor(PAPER)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
    for side in ax.spines.values():
        side.set_visible(False)
axes[0].legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="lower left")
fig.suptitle("Unseen-object fit: the best code found for each of 225 Thingi objects none of the models was trained on "
             "(dots: means; bars: 95% intervals)", fontsize=11.5, color=INK, x=0.01, ha="left")
fig.tight_layout()
fig.savefig(chart_path, dpi=115, facecolor=PAPER)
print("saved", chart_path)

# ----------------------------------------------------------------- shapes
picks = [order[int(q * (len(order) - 1))] for q in np.linspace(0.06, 0.94, 8)]
RES = 96
axis = torch.linspace(-1, 1, RES, device="cuda")
grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(1, -1, 3)


def draw(ax, vol, half):
    ax.set_facecolor(PAPER)
    ax.set_axis_off()
    if vol.max() > 0 and vol.min() < 0:
        verts, faces, _, _ = measure.marching_cubes(vol, 0.0, spacing=(2 / (RES - 1),) * 3)
        verts = verts - 1.0
        tri = verts[faces]
        normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        normal /= np.linalg.norm(normal, axis=1, keepdims=True).clip(1e-12)
        if (normal * (tri.mean(1) - verts.mean(0))).sum() < 0:
            normal = -normal
        shade = 0.22 + 0.78 * np.clip(normal @ LIGHT, 0, 1)
        ax.add_collection3d(Poly3DCollection(tri, facecolors=np.clip(BASE[None] * shade[:, None] * 1.08, 0, 1), edgecolors="none"))
    ax.set_xlim(-half, half); ax.set_ylim(-half, half); ax.set_zlim(-half, half)
    ax.set_box_aspect((1, 1, 1), zoom=1.3)
    ax.view_init(elev=22, azim=-58)


cols = 1 + len(results)
fig = plt.figure(figsize=(2.05 * cols + 0.6, 2.1 * len(picks) + 0.8))
fig.patch.set_facecolor(PAPER)
volumes = {}
for m, r in enumerate(results):
    trainer = Trainer.from_checkpoint(r["ckpt"], device="cuda", tensorboard=False)
    model = trainer.model.eval()
    for p in picks:
        code = torch.nn.functional.normalize(r["codes"][p].float().cuda(), dim=-1)[None]
        with torch.no_grad():
            vol = torch.cat([model(code, grid[:, s:s + 131072]) for s in range(0, grid.shape[1], 131072)], dim=1)
        volumes[(m, p)] = vol.reshape(RES, RES, RES).cpu().numpy()
    del trainer, model
    torch.cuda.empty_cache()
for row, p in enumerate(picks):
    target = pool["grid"][index[p]].float().numpy()
    inside = np.argwhere(target > 0)
    half = max(float(np.abs(inside / (RES - 1) * 2 - 1).max()) * 1.05, 0.3) if len(inside) else 0.9
    ax = fig.add_subplot(len(picks), cols, row * cols + 1, projection="3d")
    draw(ax, target, half)
    name = pool["names"][index[p]][:24]
    ax.set_title(f"{name}\nthickness {thick[p]:.2f}", fontsize=8, color=INK, pad=0)
    for m, r in enumerate(results):
        ax = fig.add_subplot(len(picks), cols, row * cols + 2 + m, projection="3d")
        draw(ax, volumes[(m, p)], half)
        ax.set_title((f"{r['label']}\n" if row == 0 else "") + f"distance {r['dist'][p]:.3f}", fontsize=8, color=INK if row else INK, pad=0)
fig.suptitle("Unseen objects (left) and each model's best attempt; rows ordered from thinnest to thickest; "
             "distance = mean surface distance (lower is better)", fontsize=11, color=INK, x=0.01, ha="left")
fig.subplots_adjust(left=0.01, right=0.995, top=0.95, bottom=0.005, wspace=0.0, hspace=0.12)
fig.savefig(shapes_path, dpi=100, facecolor=PAPER)
print("saved", shapes_path)
