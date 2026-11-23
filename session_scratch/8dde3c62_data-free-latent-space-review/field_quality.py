"""How accurate is the field, and does it have slope 1, at the anchor codes and at random codes?

usage: field_quality.py <checkpoint> [<checkpoint> ...]

4,096 points in the ball per code. For the anchor codes and for 32 random codes:
 - slope: |gradient| should be 1 everywhere (the eikonal condition); mean | |g| - 1 | and the trainer's (|g| - 1)^2,
   over all points and within 0.1 of the surface;
 - consistency: stepping from a point by the field value along the gradient should land on the surface;
   the miss |f(x - f g)|, mean and median;
 - (anchor codes only) the field against the anchor object's own distance field: mean |f - target|.
"""
import sys

import torch

from arm_planner.data_free import losses as L
from arm_planner.data_free.field import field_and_grad
from arm_planner.data_free.train import Trainer

for path in sys.argv[1:]:
    trainer = Trainer.from_checkpoint(path, device="cuda", tensorboard=False)
    cfg, model = trainer.cfg, trainer.model.eval()
    gen = torch.Generator().manual_seed(8)
    x = torch.randn(4096, 3, generator=gen)
    x = (x / x.norm(dim=-1, keepdim=True) * float(cfg.bound_radius) * torch.rand(4096, 1, generator=gen) ** (1 / 3)).cuda()
    sets = [("random codes", torch.nn.functional.normalize(torch.randn(32, cfg.latent_dim, generator=gen), dim=-1).cuda(), None)]
    if trainer.anchor_codes is not None:
        sets.insert(0, ("anchor codes", torch.nn.functional.normalize(trainer.anchor_codes.detach(), dim=-1), trainer.anchor_grid.float()))
    print(f"\n{path.split('/')[-2]} (update {trainer.step_count})")
    for label, codes, grids in sets:
        rows = {"slope": [], "near": [], "sq": [], "miss": [], "target": [], "f": []}
        for s in range(0, codes.shape[0], 8):
            block = codes[s:s + 8]
            pts = x[None].expand(block.shape[0], -1, -1)
            mod = model.prepare(block)
            f, g = field_and_grad(model, mod, pts, create_graph=False)
            f, g = f.detach(), g.detach()
            with torch.no_grad():
                size = g.norm(dim=-1)
                landed = model.query(mod, pts - f[..., None] * g)
                rows["slope"].append((size - 1).abs()); rows["sq"].append((size - 1).square())
                rows["miss"].append(landed.abs()); rows["f"].append(f)
                if grids is not None:
                    rows["target"].append((f - L.sample_grid(grids[s:s + 8], pts)).abs())
        f = torch.cat(rows["f"]); slope = torch.cat(rows["slope"]); near = f.abs() < 0.1
        line = (f"  {label:13s} slope error mean {float(slope.mean()):.3f} (within 0.1 of the surface {float(slope[near].mean()):.3f}; "
                f"share of points off by more than 0.2: {float((slope > 0.2).float().mean()):.3f}); trainer's eikonal term {float(torch.cat(rows['sq']).mean()):.4f}; "
                f"consistency miss mean {float(torch.cat(rows['miss']).mean()):.4f}, median {float(torch.cat(rows['miss']).median()):.4f}")
        if rows["target"]:
            line += f"; field against the object's true distance: mean error {float(torch.cat(rows['target']).mean()):.4f}"
        print(line)
