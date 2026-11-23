"""How many units of the last hidden layer are dead, and where do their inputs sit?

usage: dead_units_why.py <checkpoint> [<checkpoint> ...]
For the last hidden layer each unit's input to the activation is  V * (1 + scale) + shift, where V is
W h + b (layer-normalised if the model uses layer norm there). A unit is "on" where that input is above
-0.075 (the plain softplus gives 0.01 there) and dead if it is on for under 0.1% of 32 codes x 4,096 points.
Works for any activation or normalisation setting (it uses the decoder's own forward pass).
"""
import sys

import torch
import torch.nn.functional as F

from arm_planner.data_free.train import Trainer

for path in sys.argv[1:]:
    trainer = Trainer.from_checkpoint(path, device="cuda", tensorboard=False)
    cfg, model = trainer.cfg, trainer.model.eval()
    gen = torch.Generator().manual_seed(23)
    codes = F.normalize(torch.randn(32, cfg.latent_dim, generator=gen), dim=-1).cuda()
    x = torch.randn(4096, 3, generator=gen)
    x = (x / x.norm(dim=-1, keepdim=True) * float(cfg.bound_radius) * torch.rand(4096, 1, generator=gen) ** (1 / 3)).cuda()
    x = x[None].expand(32, -1, -1)
    with torch.no_grad():
        mod = model.prepare(codes)
        pres = []
        for start in range(0, 32, 8):
            seen = []
            model._net(mod.index(torch.arange(start, start + 8, device="cuda")), x[start:start + 8], collect=seen)
            pres.append(seen[-1])
        pre = torch.cat(pres)                                              # 32 x 4096 x W
        last = len(model.hidden) - 1
        gain, shift = 1.0 + mod.scale[:, last], mod.shift[:, last]
        on = (pre > -0.0754).float().mean(dim=(0, 1))
        dead = on < 1e-3
        out_w = model.out.weight.reshape(-1)
        print(f"\n{path.split('/')[-2]} (update {trainer.step_count}): last hidden layer, {int(dead.sum())} dead units and {int((~dead).sum())} working ones"
              f" (leak {getattr(cfg, 'activation_leak', 0.0)}, layer-normalised layers {getattr(cfg, 'layer_norm_layers', 0)})")
        for label, pick in (("dead", dead), ("working", ~dead)):
            if int(pick.sum()) == 0:
                continue
            print(f"  {label:8s} input to the activation: mean {float(pre[..., pick].mean()):+.2f} (largest seen {float(pre[..., pick].max()):+.2f}); "
                  f"multiplier (1 + scale): mean {float(gain[:, pick].mean()):+.2f}; shift: mean {float(shift[:, pick].mean()):+.2f}; "
                  f"|final-layer weight|: {float(out_w[pick].abs().mean()):.4f}")
