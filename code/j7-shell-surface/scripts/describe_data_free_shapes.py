"""Geometric descriptors of the shapes a trained model decodes from random codes.

usage: describe_data_free_shapes.py <checkpoint> <tag> [codes] [resolution]
Decodes fresh random codes on a grid, measures scripts/data_free_shape_descriptors.py on each solid (after
resizing every shape to the same mass size, so that a model whose shapes have grown is not read as thicker),
prints the 10/50/90% values of each descriptor and writes <tag>.json to the folder of this script's caller.
"""
import json
import multiprocessing as mp
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_free_shape_descriptors as SD  # noqa: E402

from arm_planner.data_free.train import Trainer  # noqa: E402


def one(job):
    grid, seed = job
    return SD.describe(grid, seed=seed)


def main():
    ckpt, tag = sys.argv[1], sys.argv[2]
    count = int(sys.argv[3]) if len(sys.argv) > 3 else 48
    res = int(sys.argv[4]) if len(sys.argv) > 4 else 80
    trainer = Trainer.from_checkpoint(ckpt, device="cuda", tensorboard=False)
    cfg, model = trainer.cfg, trainer.model.eval()
    gen = torch.Generator().manual_seed(77)
    codes = torch.nn.functional.normalize(torch.randn(count, cfg.latent_dim, generator=gen), dim=-1).cuda()
    near = float(sys.argv[5]) if len(sys.argv) > 5 else None
    if near is not None:
        # codes at a fixed cosine from the anchor codes (anchor k % K), in a random direction around each
        anchors = torch.nn.functional.normalize(trainer.anchor_codes.detach(), dim=-1)
        base = anchors[torch.arange(count, device="cuda") % anchors.shape[0]]
        side = codes - (codes * base).sum(-1, keepdim=True) * base
        side = torch.nn.functional.normalize(side, dim=-1)
        codes = near * base + (1.0 - near * near) ** 0.5 * side
    axis = torch.linspace(-1, 1, res, device="cuda")
    pts = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(1, -1, 3)
    target = float(cfg.cov_trace_target)
    grids = []
    with torch.no_grad():
        for c in codes:
            f = model(c[None], pts)[0]
            w = torch.sigmoid(f / float(cfg.soft_tau))
            centre = (w[:, None] * pts[0]).sum(0) / w.sum().clamp_min(1e-6)
            trace = (w * (pts[0] - centre).square().sum(-1)).sum() / w.sum().clamp_min(1e-6)
            s = float((target / trace.clamp_min(1e-9)).sqrt())            # resize to the target mass size
            g = s * model(c[None], pts / s + centre)[0]
            grids.append(g.reshape(res, res, res).cpu().numpy())
    del model, trainer
    torch.cuda.empty_cache()
    with mp.get_context("spawn").Pool(min(12, max(1, (os.cpu_count() or 2) - 2))) as pool:
        out = pool.map(one, [(g, k) for k, g in enumerate(grids)], chunksize=1)
    good = [d for d in out if d is not None]
    print(f"{tag}: {len(good)} of {count} random codes gave a solid; 10% / median / 90% of each descriptor")
    summary = {}
    for key in SD.KEYS:
        v = np.array([d[key] for d in good])
        q = np.quantile(v, [0.1, 0.5, 0.9])
        summary[key] = [float(x) for x in q]
        print(f"   {key:14s} {q[0]:6.2f} {q[1]:6.2f} {q[2]:6.2f}")
    extra = {"not star-shaped (under 90% of rays cross once)": float(np.mean([d["star_fraction"] < 0.9 for d in good])),
             "folded (fold above 1.5)": float(np.mean([d["fold"] > 1.5 for d in good])),
             "branched (2 or more)": float(np.mean([d["branches"] >= 2 for d in good])),
             "nearly a handle (a gap that closes)": float(np.mean([d["gap"] > 0 for d in good])),
             "mostly rod-like (rod share largest)": float(np.mean([d["linear"] > max(d["planar"], d["round"]) for d in good])),
             "mostly sheet-like": float(np.mean([d["planar"] > max(d["linear"], d["round"]) for d in good])),
             "mostly blob-like": float(np.mean([d["round"] >= max(d["linear"], d["planar"]) for d in good]))}
    for k, v in extra.items():
        print(f"   share {k}: {v:.2f}")
    json.dump({"checkpoint": ckpt, "count": count, "solids": len(good), "quantiles": summary, "shares": extra, "each": good},
              open(os.path.join(os.getcwd(), tag + "_descriptors.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
