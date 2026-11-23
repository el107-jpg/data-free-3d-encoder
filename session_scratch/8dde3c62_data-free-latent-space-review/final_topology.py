"""Topology of shapes decoded from fresh random codes (not the training probes) on a fine grid.

usage: final_topology.py <checkpoint> [count] [resolution]
Flagged shapes (more than one piece, a cavity or a handle) are recounted on finer grids so that
coarse-grid artefacts are not reported as defects.
"""
import sys

import torch

from arm_planner.data_free import diagnostics as D
from arm_planner.data_free.train import Trainer

ckpt = sys.argv[1]
count = int(sys.argv[2]) if len(sys.argv) > 2 else 128
res = int(sys.argv[3]) if len(sys.argv) > 3 else 96
chunk = int(sys.argv[4]) if len(sys.argv) > 4 else 8
trainer = Trainer.from_checkpoint(ckpt, device="cuda", tensorboard=False)
cfg, model = trainer.cfg, trainer.model.eval()
gen = torch.Generator().manual_seed(12345)
codes = torch.nn.functional.normalize(torch.randn(count, cfg.latent_dim, generator=gen), dim=-1).to("cuda")

rows = []
with torch.no_grad():
    for c0 in range(0, count, chunk):
        rows += D.decoder_grid_topology(model, codes[c0:c0 + chunk], cfg, resolution=res)
pieces = torch.tensor([r["pieces"] for r in rows])
cav = torch.tensor([r["cavities"] for r in rows])
han = torch.tensor([r["handles"] for r in rows])
flag = (pieces != 1) | (cav > 0) | (han > 0)
print(f"update {trainer.step_count}: {count} fresh random codes at resolution {res}")
print(f"  one piece {float((pieces == 1).float().mean()):.3f}  no cavity {float((cav == 0).float().mean()):.3f}  "
      f"no handle {float((han == 0).float().mean()):.3f}  clean {float((~flag).float().mean()):.3f}")
confirmed = 0
for i in torch.nonzero(flag).flatten().tolist():
    row, worst = [], True
    for r2 in (128, 160):
        with torch.no_grad():
            t = D.grid_topology(lambda p, i=i: model(codes[i:i + 1], p[None])[0], resolution=r2, device="cuda")
        row.append(f"{r2}: pieces {t['pieces']} cavities {t['cavities']} handles {t['handles']}")
        worst = worst and (t["pieces"] != 1 or t["cavities"] > 0 or t["handles"] > 0)
    confirmed += int(worst)
    print(f"  code {i:3d} | {res}: pieces {int(pieces[i])} cavities {int(cav[i])} handles {int(han[i])} | " + " | ".join(row))
print(f"  flagged {int(flag.sum())}, still defective at 128 and 160: {confirmed}  ->  "
      f"clean on fine grids {1 - confirmed / count:.3f}")
