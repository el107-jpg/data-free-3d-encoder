"""Which codes carry the field-accuracy error: the anchors' or the random ones?

For a saved model with anchors: the consistency error (mean |f at the one-step surface guess|) and the
eikonal error, per code, for the anchor codes and for fresh random codes at the same points.
usage: anchor_strain.py <checkpoint>
"""
import sys

import torch

from arm_planner.data_free.field import evaluate_field, sample_points
from arm_planner.data_free.train import Trainer

trainer = Trainer.from_checkpoint(sys.argv[1], device="cuda", tensorboard=False)
cfg, model = trainer.cfg, trainer.model
gen = torch.Generator().manual_seed(77)
anchors = torch.nn.functional.normalize(trainer.anchor_codes.detach(), dim=-1)
random = torch.nn.functional.normalize(torch.randn(anchors.shape[0], cfg.latent_dim, generator=gen), dim=-1).cuda()
x = sample_points(cfg, 1, gen, "cuda")
out = {}
for name, codes in (("anchor codes", anchors), ("random codes", random)):
    cons, eik, near = [], [], []
    for s in range(0, codes.shape[0], 8):
        block = codes[s:s + 8]
        fb = evaluate_field(model, block, torch.zeros(block.shape[0], dtype=torch.long, device="cuda"), x, cfg,
                            create_graph=False)
        cons.append(fb.fp.detach().abs().mean(dim=1))
        eik.append((fb.g.detach().norm(dim=-1) - 1.0).square().mean(dim=1))
        band = fb.f.detach().abs() < 0.1
        near.append((fb.fp.detach().abs() * band).sum(dim=1) / band.sum(dim=1).clamp_min(1))
        del fb
    cons, eik, near = torch.cat(cons), torch.cat(eik), torch.cat(near)
    out[name] = cons
    print(f"{name}: consistency error mean {float(cons.mean()):.4f} (median {float(cons.median()):.4f}, max {float(cons.max()):.4f}); "
          f"near the surface only {float(near.mean()):.4f}; eikonal {float(eik.mean()):.5f}")
print(f"update {trainer.step_count}; tolerance {cfg.tolerances['consistency']}; "
      f"a batch with 4 anchors in 32 codes would average "
      f"{float((4 * out['anchor codes'].mean() + 28 * out['random codes'].mean()) / 32):.4f}")
