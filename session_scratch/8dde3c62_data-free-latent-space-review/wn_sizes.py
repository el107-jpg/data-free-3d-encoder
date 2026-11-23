"""Sizes inside the main network: per-layer weight-row lengths, and the scales and shifts the code hands over.

usage: wn_sizes.py <checkpoint> [<checkpoint> ...]
"""
import sys

import torch

from arm_planner.data_free.train import Trainer

for path in sys.argv[1:]:
    trainer = Trainer.from_checkpoint(path, device="cpu", tensorboard=False)
    cfg, model = trainer.cfg, trainer.model.eval()
    gen = torch.Generator().manual_seed(5)
    codes = torch.nn.functional.normalize(torch.randn(64, cfg.latent_dim, generator=gen), dim=-1)
    with torch.no_grad():
        mod = model.prepare(codes)
        rows = [float(lin.weight.norm(dim=1).mean()) for lin in model.hidden]
        gain = [float((1 + mod.scale[:, k]).abs().mean()) for k in range(len(model.hidden))]
        shift = [float(mod.shift[:, k].abs().mean()) for k in range(len(model.hidden))]
        bias = [float(lin.bias.abs().mean()) for lin in model.hidden]
    print(f"{path.split('/')[-2][:28]} @ {trainer.step_count}")
    print("  hidden layer:                 " + "".join(f"{k + 1:>8d}" for k in range(len(rows))))
    print("  weight-row length (start 1.41)" + "".join(f"{v:8.2f}" for v in rows))
    print("  code's gain |1 + scale|       " + "".join(f"{v:8.2f}" for v in gain))
    print("  code's shift, typical size    " + "".join(f"{v:8.2f}" for v in shift))
    print("  bias, typical size            " + "".join(f"{v:8.2f}" for v in bias))
    print(f"  output layer: weight length {float(model.out.weight.norm()):.4f}, bias {float(model.out.bias):.4f}")
