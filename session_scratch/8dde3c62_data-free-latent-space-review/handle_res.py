"""Are the counted handles real? Recount at finer grid resolutions."""
import sys

import torch

from arm_planner.data_free import diagnostics as D
from arm_planner.data_free.train import Trainer

for label, ckpt in zip(sys.argv[1::2], sys.argv[2::2]):
    trainer = Trainer.from_checkpoint(ckpt, device="cuda", tensorboard=False)
    cfg, model = trainer.cfg, trainer.model.eval()
    codes = trainer.bank.codes[:96]
    coarse = D.decoder_grid_topology(model, codes, cfg, resolution=64)
    flagged = [i for i, t in enumerate(coarse) if t["handles"] > 0 or t["pieces"] > 1]
    print(f"== {label}: {len(flagged)} of 96 codes flagged at resolution 64")
    for i in flagged[:8]:
        row = []
        for res in (64, 96, 128, 160):
            t = D.grid_topology(lambda p, i=i: model(codes[i:i + 1], p[None])[0], resolution=res, device="cuda")
            row.append(f"{res}: pieces {t['pieces']} handles {t['handles']}")
        print(f"  code {i:3d} | " + " | ".join(row))
    del trainer
    torch.cuda.empty_cache()
