"""Where does a push for more different shapes land in the network, layer by layer?

usage: grad_by_layer.py <checkpoint> [<checkpoint> ...]

A light stand-in for the trainer's variety push: 16 random codes, 2,048 points in the ball, and the objective
-mean log(mean |f_a - f_b|) over all pairs of codes (make every pair's fields more different). One backward
pass; the share of the squared gradient in each layer, the gradient per weight, and the sizes that explain it.
"""
import sys

import torch

from arm_planner.data_free.train import Trainer

for path in sys.argv[1:]:
    trainer = Trainer.from_checkpoint(path, device="cuda", tensorboard=False)
    cfg, model = trainer.cfg, trainer.model
    gen = torch.Generator().manual_seed(17)
    codes = torch.nn.functional.normalize(torch.randn(16, cfg.latent_dim, generator=gen), dim=-1).cuda()
    x = torch.randn(2048, 3, generator=gen)
    x = (x / x.norm(dim=-1, keepdim=True) * float(cfg.bound_radius) * torch.rand(2048, 1, generator=gen) ** (1 / 3)).cuda()
    seen = {}
    hook = model.out.register_forward_hook(lambda mod, inp, out: seen.__setitem__("h", inp[0].detach()))
    f = model(codes, x[None].expand(16, -1, -1))
    hook.remove()
    gap = (f[:, None] - f[None]).abs().mean(-1)
    pairs = torch.triu(torch.ones(16, 16, dtype=torch.bool, device="cuda"), 1)
    objective = -gap[pairs].log().mean()
    model.zero_grad(set_to_none=True)
    objective.backward()
    groups = [("side network, layer 1 (64 to 256)", [model.mapping[0]]), ("side network, layer 2 (256 to 256)", [model.mapping[2]]),
              ("side network, last layer (to scales and shifts)", [model.film])]
    groups += [(f"main network, layer {k + 1}", [lin]) for k, lin in enumerate(model.hidden)]
    groups += [("main network, final layer (512 to 1)", [model.out])]
    energy = [sum(float(p.grad.square().sum()) for m in mods for p in m.parameters()) for _, mods in groups]
    count = [sum(p.numel() for m in mods for p in m.parameters()) for _, mods in groups]
    total = sum(energy)
    print(f"\n{path.split('/')[-2]} (update {trainer.step_count}); mean pair difference {float(gap[pairs].mean()):.4f}")
    print(f"  {'layer':50s} {'weights':>10s} {'share of gradient':>18s} {'typical gradient per weight':>28s}")
    for (name, _), e, n in zip(groups, energy, count):
        print(f"  {name:50s} {n:10d} {e / total:18.4f} {(e / n) ** 0.5:28.2e}")
    w, h = model.out.weight.detach().reshape(-1), seen["h"]
    print(f"  final layer: typical weight {float(w.abs().mean()):.4f}; typical input to it (last hidden value) {float(h.abs().mean()):.3f}; "
          f"ratio input/weight {float(h.abs().mean() / w.abs().mean()):.0f}")
    print(f"  share of last-hidden units that are switched off (value < 0.01) at a typical point: {float((h < 0.01).float().mean()):.2f}")
