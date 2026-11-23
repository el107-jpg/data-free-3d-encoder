"""Did a tidy-up stage repair the field without moving the shapes?

usage: tidy_compare.py <checkpoint before> <checkpoint after>
Same 48 fresh random codes and the same points for both models.
 - field accuracy: the consistency miss |f(x - f g)| over all points, for starting points near the surface
   (|f| < 0.1) and far from it, and its median (what the graph uses); the eikonal error (|g| - 1)^2;
 - shape change: overlap of the before and after solids, and how far the field moved at the old surface.
"""
import sys

import torch

from arm_planner.data_free.field import evaluate_field, sample_points
from arm_planner.data_free.train import Trainer

before_path, after_path = sys.argv[1], sys.argv[2]
gen = torch.Generator().manual_seed(99)
models = {}
for name, path in (("before", before_path), ("after", after_path)):
    t = Trainer.from_checkpoint(path, device="cuda", tensorboard=False, overrides=["consistency_fixed_surface=False",
                                                                                   "field_moves_surface=False"])
    models[name] = (t.model.eval(), t.cfg, t.step_count)
cfg = models["before"][1]
codes = torch.nn.functional.normalize(torch.randn(48, cfg.latent_dim, generator=gen), dim=-1).cuda()
x = sample_points(cfg, 1, gen, "cuda")
stats, fields = {}, {}
for name, (model, mcfg, step) in models.items():
    miss, slope, start = [], [], []
    for s in range(0, codes.shape[0], 8):
        block = codes[s:s + 8]
        fb = evaluate_field(model, block, torch.zeros(block.shape[0], dtype=torch.long, device="cuda"), x, mcfg,
                            create_graph=False)
        miss.append(fb.fp.detach().abs()[:, 1:])
        slope.append((fb.g.detach().norm(dim=-1) - 1.0).square()[:, 1:])
        start.append(fb.f.detach()[:, 1:])
        del fb
    miss, slope, start = torch.cat(miss), torch.cat(slope), torch.cat(start)
    fields[name] = start
    near = start.abs() < 0.1
    stats[name] = {
        "update": step,
        "consistency miss, all points": float(miss.mean()),
        "consistency miss, median (graph uses this; full mode needs < 0.02)": float(miss.median()),
        "consistency miss, starting within 0.1 of the surface": float(miss[near].mean()),
        "consistency miss, starting further away": float(miss[~near].mean()),
        "eikonal error, all points": float(slope.mean()),
        "eikonal error, within 0.1 of the surface": float(slope[near].mean()),
        "eikonal error, further away": float(slope[~near].mean()),
    }
print(f"{'':66s} {'before':>10s} {'after':>10s}")
for key in stats["before"]:
    a, b = stats["before"][key], stats["after"][key]
    print(f"{key:66s} {a:10.5f} {b:10.5f}" if key != "update" else f"{key:66s} {a:10d} {b:10d}")
fa, fb_ = fields["before"], fields["after"]
inside_a, inside_b = fa > 0, fb_ > 0
iou = (inside_a & inside_b).sum(1).float() / (inside_a | inside_b).sum(1).clamp_min(1).float()
band = fa.abs() < 0.02
shift = ((fb_ - fa).abs() * band).sum(1) / band.sum(1).clamp_min(1)
print(f"\nshape change over 48 codes: overlap of before and after solids mean {float(iou.mean()):.4f} (worst {float(iou.min()):.4f}); "
      f"field moved at the old surface by {float(shift.mean()):.4f} on average (worst code {float(shift.max()):.4f})")
