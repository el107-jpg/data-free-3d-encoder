"""How different are the shapes at random codes from one another? Global measures that do not use the code's
local derivative, plus how thin the shapes are.

usage: variety_pairs.py <out.json> <label>=<checkpoint> [...]

96 fresh random codes (the same for every model). For every pair of shapes:
  - overlap distance: 1 - (volume in both) / (volume in either), on 30,000 points in the ball;
  - surface distance: mean distance from a surface point of one shape to the nearest surface point of the
    other, averaged both ways (1,500 surface points per shape, slid onto the surface with Newton steps);
  - field distance: mean |f_a - f_b| over the ball (what the training's variety push uses).
Reported: mean over all pairs, and mean distance to the nearest other shape. No pose is removed (all models
are trained to the same centre and size). Per shape: share of the ball it fills, surface area, compactness
(36 pi V^2 / A^3: 1 for a ball, small for thin or branching shapes) and half-thickness (largest depth inside,
found by sliding; the field's own value at its deepest point).
"""
import json
import sys

import numpy as np
import torch
from skimage import measure

from arm_planner.data_free.train import Trainer

N, POINTS, SURF, RES = 96, 30000, 1500, 72
out = {}
for item in sys.argv[2:]:
    label, path = item.split("=", 1)
    trainer = Trainer.from_checkpoint(path, device="cuda", tensorboard=False)
    cfg, model = trainer.cfg, trainer.model.eval()
    gen = torch.Generator().manual_seed(515)
    codes = torch.nn.functional.normalize(torch.randn(N, int(cfg.latent_dim), generator=gen), dim=-1).cuda()
    x = torch.randn(POINTS, 3, generator=gen)
    x = (x / x.norm(dim=-1, keepdim=True) * 0.9 * torch.rand(POINTS, 1, generator=gen) ** (1 / 3)).cuda()
    axis = torch.linspace(-1, 1, RES, device="cuda")
    grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(1, -1, 3)
    fields, clouds, volume, area, depth = [], [], [], [], []
    for k in range(N):
        c = codes[k:k + 1]
        with torch.no_grad():
            f = model(c, x[None])[0]
            g3 = torch.cat([model(c, grid[:, s:s + 131072]) for s in range(0, grid.shape[1], 131072)], dim=1).reshape(RES, RES, RES).cpu().numpy()
        fields.append(f)
        cell = 2.0 / (RES - 1)
        volume.append(float((g3 > 0).sum()) * cell ** 3)
        if g3.max() > 0 and g3.min() < 0:
            verts, faces, _, _ = measure.marching_cubes(g3, 0.0, spacing=(cell,) * 3)
            area.append(float(measure.mesh_surface_area(verts, faces)))
        else:
            area.append(float("nan"))
        depth.append(float(g3.max()))
        # surface points: start from the ball samples closest to the surface, slide on
        p = x[f.abs().topk(SURF, largest=False).indices][None]
        for _ in range(5):
            p = p.detach().requires_grad_(True)
            v = model(c, p)
            (gr,) = torch.autograd.grad(v.sum(), p)
            p = (p - (v.detach()[..., None] * gr / gr.square().sum(-1, keepdim=True).clamp_min(1e-4)).clamp(-0.05, 0.05)).detach()
        clouds.append(p[0])
    F = torch.stack(fields)
    inside = F > 0
    both = (inside[:, None] & inside[None]).sum(-1).float()
    either = (inside[:, None] | inside[None]).sum(-1).float().clamp_min(1)
    d_iou = (1 - both / either).cpu().numpy()
    d_field = torch.stack([(F[i][None] - F).abs().mean(-1) for i in range(N)]).cpu().numpy()
    d_surf = np.zeros((N, N))
    for i in range(N):
        d = torch.cdist(clouds[i][None].expand(N, -1, -1), torch.stack(clouds))      # N x S x S
        d_surf[i] = (0.5 * (d.min(dim=2).values.mean(1) + d.min(dim=1).values.mean(1))).cpu().numpy()
    off = ~np.eye(N, dtype=bool)
    res = {"update": int(trainer.step_count)}
    for name, d in (("overlap", d_iou), ("surface", d_surf), ("field", d_field)):
        nearest = np.where(off, d, np.inf).min(1)
        res[name] = {"mean_pair": float(d[off].mean()), "nearest_mean": float(nearest.mean()),
                     "nearest_se": float(nearest.std(ddof=1) / np.sqrt(N))}
    volume, area, depth = np.array(volume), np.array(area), np.array(depth)
    compact = 36 * np.pi * volume ** 2 / area ** 3
    ball = 4 / 3 * np.pi * 0.9 ** 3
    res["shape"] = {"fill_mean": float(np.mean(volume / ball)), "area_mean": float(np.nanmean(area)),
                    "compactness_median": float(np.nanmedian(compact)), "compactness_p10": float(np.nanpercentile(compact, 10)),
                    "half_thickness_median": float(np.median(depth)), "half_thickness_p10": float(np.percentile(depth, 10)),
                    "volume_rel_std": float(volume.std() / volume.mean()), "area_rel_std": float(np.nanstd(area) / np.nanmean(area)),
                    "compactness_std": float(np.nanstd(compact))}
    out[label] = res
    s = res["shape"]
    print(f"\n{label} (update {res['update']})")
    for name, unit in (("overlap", "1 - overlap"), ("surface", "surface distance"), ("field", "field difference")):
        r = res[name]
        print(f"  {unit:17s}: mean over pairs {r['mean_pair']:.4f}; to the nearest other shape {r['nearest_mean']:.4f} +/- {r['nearest_se']:.4f}")
    print(f"  shapes: fill {s['fill_mean']:.3f} of the ball; surface area {s['area_mean']:.2f}; compactness median {s['compactness_median']:.2f} "
          f"(thinnest tenth {s['compactness_p10']:.2f}); half-thickness median {s['half_thickness_median']:.3f} (thinnest tenth {s['half_thickness_p10']:.3f}); "
          f"spread between shapes: volume {s['volume_rel_std']:.2f}, area {s['area_rel_std']:.2f}, compactness {s['compactness_std']:.2f}", flush=True)
    del trainer, model
    torch.cuda.empty_cache()
json.dump(out, open(sys.argv[1], "w"), indent=1)
