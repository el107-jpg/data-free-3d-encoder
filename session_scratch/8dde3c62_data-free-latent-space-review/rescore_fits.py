"""Re-score saved unseen-object fits with measures that do not depend on how thick the object is.

usage: rescore_fits.py <checkpoint> <fit tag> [<worst-objects picture.png>]
Reads <fit tag>.json / .pt (heldout_fit.py: the best code found for each unseen object); no new search.

For every object, the surface of the object and the surface of the decoder's best attempt are each turned into
points (marching cubes on a 96-cell grid) and compared:
  surface distance   mean distance from each surface to the other, averaged over both directions
  missed             95th percentile of the distance from the OBJECT's surface to the attempt (parts not reached)
  extra              95th percentile of the distance from the ATTEMPT's surface to the object (parts added)
  F-score at t       share of each surface lying within t of the other (harmonic mean of the two shares)
  pieces             number of separate pieces of the attempt (the object is always one piece)
Distances are in the units of the box (a typical object is about 1 across). Writes <fit tag>_surface.json.
"""
import json
import multiprocessing as mp
import os
import sys

import numpy as np
import scipy.ndimage as ndi
import torch
from scipy.spatial import cKDTree
from skimage import measure

sys.path.insert(0, "scripts")
import data_free_shape_descriptors as SD  # noqa: E402

from arm_planner.data_free.train import Trainer  # noqa: E402

ckpt, tag = sys.argv[1], sys.argv[2]
picture = sys.argv[3] if len(sys.argv) > 3 else ""
D = "/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/analysis_scripts/followup/"
pool = torch.load("/home/eric-luo/Downloads/pointcloud-motion-planning/datasets/data_free_anchors_20261112_v3/eligible.pt",
                  map_location="cpu", weights_only=False)
fit = json.load(open(D + tag + ".json"))["held_out"]
saved = torch.load(D + tag + ".pt", map_location="cpu", weights_only=False)
index, codes = saved["held_index"], saved["held_codes"]
iou = np.array(fit["per_object_iou"])
names = [str(pool["names"][k]).replace("&#039;", "'").replace("&amp;", "&")[:26] for k in index]
grids = pool["grid"]
RES = 96


def surface(grid):
    grid = np.asarray(grid, np.float32)
    if grid.max() <= 0 or grid.min() >= 0:
        return np.zeros((0, 3))
    verts, _, _, _ = measure.marching_cubes(grid, level=0.0)
    return verts / (grid.shape[0] - 1) * 2.0 - 1.0


def describe(k):
    return SD.describe(grids[k].float().numpy(), seed=int(k))


if __name__ == "__main__":
    cache = D + "heldout_descriptors_v3.json"
    if os.path.exists(cache):
        stored = json.load(open(cache))
        desc = [stored[str(k)] for k in index]
    else:
        with mp.get_context("fork").Pool(10) as workers:
            desc = workers.map(describe, index, chunksize=4)
        json.dump({str(k): d for k, d in zip(index, desc)}, open(cache, "w"))
    trainer = Trainer.from_checkpoint(ckpt, device="cuda", tensorboard=False)
    model = trainer.model.eval()
    axis = torch.linspace(-1, 1, RES, device="cuda")
    pts = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(1, -1, 3)
    rows = []
    for j, k in enumerate(index):
        with torch.no_grad():
            ours = torch.cat([model(codes[j:j + 1].cuda(), pts[:, s:s + 250_000])[0] for s in range(0, pts.shape[1], 250_000)])
        ours = ours.reshape(RES, RES, RES).cpu().numpy()
        a, b = surface(grids[k].float().numpy()), surface(ours)
        if len(b) == 0:
            rows.append({"distance": 1.0, "missed": 1.0, "extra": 1.0, "f02": 0.0, "f04": 0.0, "pieces": 0})
            continue
        to_fit, _ = cKDTree(b).query(a)
        to_obj, _ = cKDTree(a).query(b)
        f = lambda t: 2 * (to_fit < t).mean() * (to_obj < t).mean() / max(1e-9, (to_fit < t).mean() + (to_obj < t).mean())
        rows.append({"distance": float(0.5 * (to_fit.mean() + to_obj.mean())), "missed": float(np.percentile(to_fit, 95)),
                     "extra": float(np.percentile(to_obj, 95)), "f02": float(f(0.02)), "f04": float(f(0.04)),
                     "pieces": int(ndi.label(ours > 0)[1])})
    keys = ("distance", "missed", "extra", "f02", "f04", "pieces")
    m = {key: np.array([r[key] for r in rows]) for key in keys}
    json.dump({"file_ids": [int(pool["file_ids"][k]) for k in index], "names": names, "iou": iou.tolist(),
               **{key: m[key].tolist() for key in keys}}, open(D + tag + "_surface.json", "w"))
    thick = np.array([d["thickness"] for d in desc]); solid = np.array([d["solidity"] for d in desc])
    volume = np.array([float((grids[k].float() > 0).float().mean()) * 8.0 for k in index])
    star = np.array([d["star_fraction"] for d in desc])
    rank = lambda v: np.argsort(np.argsort(v))
    corr = lambda a, b: float(np.corrcoef(rank(a), rank(b))[0, 1])
    print(f"{tag}: {len(iou)} unseen objects")
    print(f"  overlap {iou.mean():.3f} | surface distance mean {m['distance'].mean():.4f}, median {np.median(m['distance']):.4f} | missed (95th pct) "
          f"{m['missed'].mean():.4f} | extra (95th pct) {m['extra'].mean():.4f} | F-score at 0.02: {m['f02'].mean():.3f}, at 0.04: {m['f04'].mean():.3f} | "
          f"attempt in one piece: {int((m['pieces'] == 1).sum())} of {len(iou)}")
    third = len(iou) // 3
    order = np.argsort(thick)
    for label, pick in (("thinnest third", order[:third]), ("middle third", order[third:2 * third]), ("thickest third", order[2 * third:])):
        print(f"  {label:15s} overlap {iou[pick].mean():.3f} | surface distance {m['distance'][pick].mean():.4f} | missed {m['missed'][pick].mean():.4f} | "
              f"extra {m['extra'][pick].mean():.4f} | F at 0.02 {m['f02'][pick].mean():.3f} | F at 0.04 {m['f04'][pick].mean():.3f} | "
              f"one piece {int((m['pieces'][pick] == 1).sum())} of {len(pick)}")
    print("  rank correlation with thickness / solidity / volume / star-shaped:")
    for label, v, sign in (("overlap", iou, 1), ("surface distance (lower is better)", m["distance"], -1), ("F-score at 0.02", m["f02"], 1),
                           ("F-score at 0.04", m["f04"], 1)):
        print(f"    {label:36s} {sign * corr(v, thick):+.2f} / {sign * corr(v, solid):+.2f} / {sign * corr(v, volume):+.2f} / {sign * corr(v, star):+.2f}"
              + ("   (sign flipped so that + means thicker fits better)" if sign < 0 else ""))
    worst = np.argsort(-m["distance"])[:16]
    print("  the 16 worst by surface distance:")
    for n in worst:
        print(f"    distance {m['distance'][n]:.3f}  missed {m['missed'][n]:.3f}  extra {m['extra'][n]:.3f}  F02 {m['f02'][n]:.2f}  pieces {m['pieces'][n]}  "
              f"overlap {iou[n]:.2f}  {names[n]:26s} thickness {thick[n]:.2f} solidity {solid[n]:.2f}")
    if picture:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from arm_planner.data_free import losses as L
        res, extent = 96, 0.9
        elev, azim = np.deg2rad(22.0), np.deg2rad(35.0)
        w = np.array([np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)])
        u = np.cross([0.0, 0.0, 1.0], w); u /= np.linalg.norm(u)
        v = np.cross(w, u)
        ax_ = torch.linspace(-extent, extent, res, device="cuda")
        a_, b_, c_ = torch.meshgrid(ax_, ax_, ax_, indexing="ij")
        frame = torch.tensor(np.stack([u, v, w]), dtype=torch.float32, device="cuda")
        view = (a_[..., None] * frame[0] + b_[..., None] * frame[1] + c_[..., None] * frame[2]).reshape(1, -1, 3)
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

        fig, axes = plt.subplots(4, 8, figsize=(17.6, 9.8), squeeze=False)
        fig.patch.set_facecolor(PAPER)
        for ax in axes.ravel():
            ax.set_axis_off()
        for n, j in enumerate(worst):
            r, c = divmod(n, 8)
            target = grids[index[j]:index[j] + 1].float().cuda()
            code = codes[j:j + 1].cuda()
            axes[2 * r, c].imshow(render(lambda p: L.sample_grid(target, p)[0], TAN))
            axes[2 * r, c].set_title(names[j], fontsize=8.5, color=INK, pad=2)
            axes[2 * r + 1, c].imshow(render(lambda p: model(code, p)[0], BLUE))
            axes[2 * r + 1, c].set_title(f"surface distance {m['distance'][j]:.3f}, pieces {m['pieces'][j]}", fontsize=8.5, color=MUTED, pad=2)
        fig.suptitle(f"The 16 unseen Thingi objects whose surface this model misses by most (tan), each above its best attempt (blue); "
                     f"model at update {trainer.step_count}", fontsize=12, color=INK, y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.975])
        fig.savefig(picture, dpi=100, facecolor=fig.get_facecolor())
        print("  saved", picture)
