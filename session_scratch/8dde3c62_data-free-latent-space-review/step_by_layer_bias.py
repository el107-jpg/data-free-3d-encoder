"""Which layers actually change the shapes in one optimiser step?

usage: step_by_layer.py <checkpoint> [<checkpoint> ...]

The checkpoint's own Adam state (momentum and second moment, current learning rate) gives the step the next
update would take. That step is applied to one group of layers at a time and the change of the field is
measured on 16 random codes x 4,096 points in the ball:
  - change: mean |delta f| (how much the shapes move);
  - between codes: mean |delta f - mean over codes of delta f| (how much the differences between shapes move;
    a change that is the same for every code does not count here).
Also the relative size of the step for each group: |step| / |weights|.
"""
import sys

import torch

from arm_planner.data_free.train import Trainer

for path in sys.argv[1:]:
    trainer = Trainer.from_checkpoint(path, device="cuda", tensorboard=False)
    cfg, model, opt = trainer.cfg, trainer.model.eval(), trainer.optimizer
    group = opt.param_groups[0]
    lr, (b1, b2), eps = group["lr"], group["betas"], group["eps"]
    step = {}
    for p in model.parameters():
        st = opt.state.get(p, {})
        if "exp_avg" not in st:
            continue
        t = float(st["step"])
        m_hat = st["exp_avg"] / (1 - b1 ** t)
        v_hat = st["exp_avg_sq"] / (1 - b2 ** t)
        step[p] = -lr * m_hat / (v_hat.sqrt() + eps)
    groups = [("side network, layers 1-2", list(model.mapping.parameters())),
              ("side network, last layer", list(model.film.parameters()))]
    groups += [(f"main network, layer {k + 1}", list(lin.parameters())) for k, lin in enumerate(model.hidden)]
    groups += [("final layer, 512 weights", [model.out.weight]), ("final layer, bias", [model.out.bias])]
    gen = torch.Generator().manual_seed(23)
    codes = torch.nn.functional.normalize(torch.randn(16, cfg.latent_dim, generator=gen), dim=-1).cuda()
    x = torch.randn(4096, 3, generator=gen)
    x = (x / x.norm(dim=-1, keepdim=True) * float(cfg.bound_radius) * torch.rand(4096, 1, generator=gen) ** (1 / 3)).cuda()
    xs = x[None].expand(16, -1, -1)
    with torch.no_grad():
        base = model(codes, xs)
        rows, total_change, total_between = [], 0.0, 0.0
        for name, params in groups:
            params = [p for p in params if p in step]
            for p in params:
                p.add_(step[p])
            delta = model(codes, xs) - base
            for p in params:
                p.sub_(step[p])
            change = float(delta.abs().mean())
            between = float((delta - delta.mean(0, keepdim=True)).abs().mean())
            size = float(torch.sqrt(sum(step[p].square().sum() for p in params)))
            norm = float(torch.sqrt(sum(p.square().sum() for p in params)))
            rows.append((name, change, between, size / max(norm, 1e-12)))
            total_change += change
            total_between += between
        for p in step:
            p.add_(step[p])
        whole = model(codes, xs) - base
        for p in step:
            p.sub_(step[p])
    print(f"\n{path.split('/')[-2][:40]} (update {trainer.step_count}); learning rate {lr:.3g}; whole step: "
          f"change {float(whole.abs().mean()):.2e}, between codes {float((whole - whole.mean(0, keepdim=True)).abs().mean()):.2e}")
    print(f"  {'layer group':28s} {'step / weights':>15s} {'change':>10s} {'share':>7s} {'between codes':>14s} {'share':>7s}")
    for name, change, between, rel in rows:
        print(f"  {name:28s} {rel:15.2e} {change:10.2e} {change / total_change:7.2f} {between:14.2e} {between / total_between:7.2f}")
