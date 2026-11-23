"""How far from the centre do shapes reach, in the trainer's pose convention?

usage: reach.py <label>=<checkpoint or grid file> [...]

A checkpoint: 96 random codes, 80,000 points in the ball of radius 1 per code. A grid file (anchors.pt or
eligible.pt; distance grids, positive inside): its objects. For each shape: reach = the largest distance from
the centre of any point inside it. Reported: median and 90th percentile of the reach, the share of shapes
reaching past 0.65 / 0.75 / 0.85, and the share of all inside volume lying past 0.5 / 0.65 / 0.75.
With the sphere base and correction bound 0.35 nothing can reach past 0.85; the correction is half way to
its bound (where it starts to flatten) at 0.675.
"""
import sys

import numpy as np
import torch

from arm_planner.data_free.train import Trainer

for item in sys.argv[1:]:
    label, path = item.split("=", 1)
    data = torch.load(path, map_location="cpu", weights_only=False)
    reach, shells = [], np.zeros(4)
    if "grid" in data:
        grids, res = data["grid"].float(), int(data["resolution"])
        keep = [i for i, c in enumerate(data.get("chosen", [False] * grids.shape[0]))] if "chosen" in data else range(grids.shape[0])
        axis = torch.linspace(-1, 1, res)
        rho = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1).norm(dim=-1)
        for i in keep:
            inside = grids[i] > 0
            r = rho[inside]
            reach.append(float(r.max()))
            shells += np.array([float(inside.sum()), float((r > 0.5).sum()), float((r > 0.65).sum()), float((r > 0.75).sum())])
        note = f"{len(reach)} objects"
    else:
        trainer = Trainer.from_checkpoint(path, device="cuda", tensorboard=False)
        cfg, model = trainer.cfg, trainer.model.eval()
        gen = torch.Generator().manual_seed(515)
        codes = torch.nn.functional.normalize(torch.randn(96, int(cfg.latent_dim), generator=gen), dim=-1).cuda()
        x = torch.randn(80000, 3, generator=gen)
        x = (x / x.norm(dim=-1, keepdim=True) * torch.rand(80000, 1, generator=gen) ** (1 / 3)).cuda()
        rho = x.norm(dim=-1)
        for k in range(96):
            with torch.no_grad():
                inside = model(codes[k:k + 1], x[None])[0] > 0
            r = rho[inside]
            reach.append(float(r.max()) if r.numel() else 0.0)
            shells += np.array([float(inside.sum()), float((r > 0.5).sum()), float((r > 0.65).sum()), float((r > 0.75).sum())])
        note = f"update {trainer.step_count}, 96 random codes"
        del trainer, model
        torch.cuda.empty_cache()
    reach = np.array(reach)
    print(f"{label} ({note}): reach median {np.median(reach):.2f}, 90th percentile {np.percentile(reach, 90):.2f}, largest {reach.max():.2f}; "
          f"shapes reaching past 0.65 / 0.75 / 0.85: {(reach > 0.65).mean():.0%} / {(reach > 0.75).mean():.0%} / {(reach > 0.85).mean():.0%}; "
          f"volume past 0.5 / 0.65 / 0.75: {shells[1] / shells[0]:.1%} / {shells[2] / shells[0]:.1%} / {shells[3] / shells[0]:.2%}", flush=True)
