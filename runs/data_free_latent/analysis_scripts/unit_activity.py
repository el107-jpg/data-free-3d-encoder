"""How many hidden units of the main network are in use, layer by layer?

usage: unit_activity.py <checkpoint> [<checkpoint> ...]

32 random codes, 4,096 points in the ball. For every hidden unit: the share of (code, point) pairs at which it
is switched on (input to the activation above -0.075, where the plain softplus gives 0.01). A unit that is on
for under 0.1% of pairs is counted as dead, one that is on for over 99.9% as always on (it then acts as a
plain linear unit). A freshly initialised network of the same shape is measured the same way.
Works for any activation or normalisation setting (it uses the decoder's own forward pass).
"""
import sys

import torch
import torch.nn.functional as F

from arm_planner.data_free.decoder import ModulatedSphereDecoder
from arm_planner.data_free.train import Trainer


def activity(model, cfg, gen):
    codes = F.normalize(torch.randn(32, cfg.latent_dim, generator=gen), dim=-1).cuda()
    x = torch.randn(4096, 3, generator=gen)
    x = (x / x.norm(dim=-1, keepdim=True) * float(cfg.bound_radius) * torch.rand(4096, 1, generator=gen) ** (1 / 3)).cuda()
    x = x[None].expand(32, -1, -1)
    rows = []
    with torch.no_grad():
        for start in range(0, 32, 8):                                      # 8 codes at a time (memory)
            seen = []
            model._net(model.prepare(codes[start:start + 8]), x[start:start + 8], collect=seen)
            for layer, pre in enumerate(seen):
                on = (pre > -0.0754).float()
                if start == 0:
                    rows.append([on.sum(dim=(0, 1)), [on.mean(dim=1)]])
                else:
                    rows[layer][0] += on.sum(dim=(0, 1))
                    rows[layer][1].append(on.mean(dim=1))
    out = []
    for total, per_code in rows:
        per_unit = total / (32 * 4096)
        per_code = torch.cat(per_code)                                     # 32 x W: share of points, per code
        out.append((float((per_unit < 1e-3).float().mean()), float((per_unit > 0.999).float().mean()), float(per_unit.mean()),
                    float(per_unit.median()), float((per_code > 1e-3).float().sum(1).mean())))
    return out


for path in sys.argv[1:]:
    trainer = Trainer.from_checkpoint(path, device="cuda", tensorboard=False)
    cfg = trainer.cfg
    torch.manual_seed(0)
    fresh = ModulatedSphereDecoder(cfg).cuda().eval()
    for label, model in ((f"{path.split('/')[-2]} (update {trainer.step_count})", trainer.model.eval()), ("  the same network, freshly initialised", fresh)):
        rows = activity(model, cfg, torch.Generator().manual_seed(23))
        print(f"\n{label}")
        print(f"  {'layer':7s} {'dead units':>11s} {'always on':>10s} {'mean share of pairs on':>23s} {'median':>8s} {'units in use per code (of ' + str(cfg.width) + ')':>32s}")
        for k, (dead, full, mean, median, per_code) in enumerate(rows):
            print(f"  {k + 1:<7d} {dead:11.2f} {full:10.2f} {mean:23.3f} {median:8.3f} {per_code:32.0f}")
