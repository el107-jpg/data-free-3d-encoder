"""Which loss term shrinks the shapes in the first updates of a run?

usage: early_collapse.py <checkpoint of the run whose settings to use> <updates> [label=key=value,key=value ...]

For each variant a NEW, untrained model with the checkpoint's settings (plus the variant's changes) is trained
for <updates> updates. Reported at 16 random codes: the share of the bounding ball that is inside the shape
(the starting sphere of radius 0.5 fills 17% of a ball of radius 0.9), the trainer's field error, and the
mean field value at the centre (0.5 for the starting sphere).
"""
import sys

import torch

from arm_planner.data_free.config import DataFreeConfig
from arm_planner.data_free.train import Trainer, apply_overrides

path, updates = sys.argv[1], int(sys.argv[2])
variants = [("as run", [])] + [(v.split("=", 1)[0], v.split("=", 1)[1].split(",")) for v in sys.argv[3:]]
state = torch.load(path, map_location="cpu", weights_only=False)
gen = torch.Generator().manual_seed(8)
x = torch.randn(20000, 3, generator=gen)
x = (x / x.norm(dim=-1, keepdim=True) * 0.9 * torch.rand(20000, 1, generator=gen) ** (1 / 3)).cuda()
print(f"{path.split('/')[-2][:50]}: {updates} updates from a fresh start")
for label, overrides in variants:
    cfg = apply_overrides(DataFreeConfig.from_dict(state["config"]), overrides)
    trainer = Trainer(cfg, device="cuda", tensorboard=False)
    codes = torch.nn.functional.normalize(torch.randn(16, cfg.latent_dim, generator=torch.Generator().manual_seed(9)), dim=-1).cuda()

    def measure():
        with torch.no_grad():
            f = trainer.model(codes, x[None].expand(16, -1, -1))
            centre = trainer.model(codes, torch.zeros(16, 1, 3, device="cuda"))
        return float((f > 0).float().mean()), float(centre.mean())

    before = measure()
    record = {}
    for _ in range(updates):
        record = trainer.step()
    after = measure()
    print(f"  {label:34s} inside share {before[0]:.3f} -> {after[0]:.3f}; field at the centre {before[1]:.2f} -> {after[1]:.2f}; "
          f"field error {record.get('e', float('nan')):.3f}; anchor term {record.get('anchor/term', float('nan')):.3f}", flush=True)
    del trainer
    torch.cuda.empty_cache()
