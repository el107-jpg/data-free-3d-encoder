"""Per-anchor overlap at the trained codes: which objects the decoder fails to reproduce.

usage: anchor_iou.py <checkpoint>
"""
import json
import os
import sys

import torch

from arm_planner.data_free import losses as L
from arm_planner.data_free.train import Trainer

trainer = Trainer.from_checkpoint(sys.argv[1], device="cuda", tensorboard=False)
cfg, model = trainer.cfg, trainer.model.eval()
meta = json.load(open(os.path.join(os.path.dirname(str(cfg.anchor_path)), "anchors.json")))["anchors"]
gen = torch.Generator().manual_seed(5)
x = (torch.rand(20000, 3, generator=gen) * 2 - 1).cuda()
codes = torch.nn.functional.normalize(trainer.anchor_codes.detach(), dim=-1)
rows = []
with torch.no_grad():
    for k in range(codes.shape[0]):
        f = model(codes[k:k + 1], x[None])[0]
        t = L.sample_grid(trainer.anchor_grid[k:k + 1].float(), x[None])[0]
        ours, theirs = f > 0, t > 0
        rows.append((float((ours & theirs).sum() / (ours | theirs).sum().clamp_min(1)), k, float(theirs.float().mean()),
                     float(ours.float().mean())))
ious = torch.tensor([r[0] for r in rows])
print(f"update {trainer.step_count}: overlap mean {float(ious.mean()):.3f} median {float(ious.median()):.3f} min {float(ious.min()):.3f}; "
      f"below 0.5: {int((ious < 0.5).sum())}, below 0.7: {int((ious < 0.7).sum())} of {len(rows)}")
for iou, k, vt, vo in sorted(rows)[:8]:
    m = meta[k]
    print(f"  anchor {k:2d} overlap {iou:.2f}  target volume share {vt:.4f}  decoded {vo:.4f}  depth at centre {m['depth_at_centre']:.3f}  "
          f"thin share {m['thin_fraction']:.3f}  {str(m['name'])[:30]}")
