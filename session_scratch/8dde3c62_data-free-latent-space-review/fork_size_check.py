"""How big are the shapes a checkpoint decodes, under its own pose rule?  usage: fork_size_check.py <checkpoint>"""
import sys

import torch

from arm_planner.data_free import losses as L
from arm_planner.data_free.train import Trainer

trainer = Trainer.from_checkpoint(sys.argv[1], device="cuda", tensorboard=False)
cfg, model = trainer.cfg, trainer.model.eval()
power, target = float(cfg.pose_power), L.pose_size_target(cfg)
gen = torch.Generator().manual_seed(5)
codes = torch.nn.functional.normalize(torch.randn(128, cfg.latent_dim, generator=gen), dim=-1).cuda()
x = (torch.rand(1, 60000, 3, generator=gen) * 2 - 1).cuda()


def measure(codes):
    out = []
    with torch.no_grad():
        for s in range(0, codes.shape[0], 8):
            f = model(codes[s:s + 8], x.expand(codes[s:s + 8].shape[0], -1, -1))
            w = torch.sigmoid(f / float(cfg.soft_tau))
            offset, moment = L.extent_pose(w, x.expand_as(torch.zeros(f.shape[0], x.shape[1], 3)), power)
            out.append(torch.stack([(f > 0).float().mean(1), moment.pow(1 / power), offset.norm(dim=-1), f.max(1).values], 1))
    return torch.cat(out).cpu()


def report(name, m):
    vol, size, off, top = m.T
    empty = vol < 1e-4
    q = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])
    print(f"{name}: {len(m)} codes; empty (under 0.01% of the box) {int(empty.sum())}; "
          f"volume fraction median {float(vol.median()):.4f}")
    if (~empty).any():
        print(f"   size / target of the non-empty ones, 10/25/50/75/90%: "
              + " ".join(f"{float(v) / target:.2f}" for v in torch.quantile(size[~empty], q))
              + f"; centre offset median {float(off[~empty].median()):.3f}")
    print(f"   deepest point of the field, 10/50/90%: " + " ".join(f"{float(v):.3f}" for v in torch.quantile(top, torch.tensor([0.1, 0.5, 0.9]))))


print(f"update {trainer.step_count}, pose_power {power:g}, size target {target:.4f}")
report("random codes", measure(codes))
if trainer.anchor_codes is not None:
    report("anchor codes", measure(torch.nn.functional.normalize(trainer.anchor_codes.detach(), dim=-1)))
