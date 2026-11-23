"""How much does one optimiser-sized nudge of the weights move the field: today's start against the fitted sphere?

usage: sphere_step_size.py <recipe config.json> <sphere_fit.pt> <fourier bands>

Adam moves every weight by about the learning rate per update. Here every weight of a layer group is moved by
+/- lr (random signs, 8 draws) and the mean |change of the field| is measured at 8 codes x 4,096 points in the
ball. Random signs understate a real, coherent step, equally for both networks, so the ratio is what counts.
 - today: f = (R - rho) + bounded correction, fresh initialisation of the recipe;
 - fitted: f = the plain network's output, weights from the sphere fit, code spread at 0.05 x the usual.
"""
import json
import sys

import torch

from arm_planner.data_free.config import DataFreeConfig
from arm_planner.data_free.decoder import ModulatedSphereDecoder, Modulation

raw = json.load(open(sys.argv[1]))
bands = int(sys.argv[3])
cfg = DataFreeConfig.from_dict(raw)
cfg.fourier_bands = bands
lr = float(cfg.lr)
dev = "cuda"
torch.manual_seed(0)
today = ModulatedSphereDecoder(cfg).to(dev)
fitted = ModulatedSphereDecoder(cfg).to(dev)
fitted.load_state_dict(torch.load(sys.argv[2], map_location=dev))
gen = torch.Generator().manual_seed(4)
codes = torch.nn.functional.normalize(torch.randn(8, int(cfg.latent_dim), generator=gen), dim=-1).to(dev)
x = torch.randn(4096, 3, generator=gen)
x = (x / x.norm(dim=-1, keepdim=True) * 0.9 * torch.rand(4096, 1, generator=gen) ** (1 / 3)).to(dev)
xs = x[None].expand(8, -1, -1)


def field_today(model):
    return model(codes, xs)


def field_fitted(model):
    mod = model.prepare(codes)
    return model._net(Modulation(mod.scale * 0.05, mod.shift * 0.05), xs)


print(f"learning rate {lr:.3g}; mean |change of the field| when every weight of a group moves by +/- that much")
print(f"  {'layer group':34s} {'today':>10s} {'fitted sphere':>14s} {'ratio':>7s}")
for name, pick in (("side network", lambda m: list(m.mapping.parameters()) + list(m.film.parameters())),
                   ("main network, hidden layers 1-8", lambda m: list(m.hidden.parameters())),
                   ("main network, final layer", lambda m: list(m.out.parameters())),
                   ("everything", lambda m: list(m.parameters()))):
    out = []
    for model, fn in ((today, field_today), (fitted, field_fitted)):
        with torch.no_grad():
            base, total = fn(model), 0.0
            for draw in range(8):
                g = torch.Generator(device=dev).manual_seed(100 + draw)
                steps = [lr * (torch.randint(0, 2, p.shape, device=dev, generator=g) * 2 - 1).to(p.dtype) for p in pick(model)]
                for p, s in zip(pick(model), steps):
                    p.add_(s)
                total += float((fn(model) - base).abs().mean()) / 8
                for p, s in zip(pick(model), steps):
                    p.sub_(s)
        out.append(total)
    print(f"  {name:34s} {out[0]:10.2e} {out[1]:14.2e} {out[1] / out[0]:7.1f}")
with torch.no_grad():
    for label, model in (("today", today), ("fitted", fitted)):
        seen = {}
        hook = model.out.register_forward_hook(lambda mod, inp, o: seen.__setitem__("h", inp[0]))
        (field_today if label == "today" else field_fitted)(model)
        hook.remove()
        print(f"  {label}: final layer length {float(model.out.weight.norm()):.3f}; last hidden layer's typical value {float(seen['h'].abs().mean()):.2f}")
