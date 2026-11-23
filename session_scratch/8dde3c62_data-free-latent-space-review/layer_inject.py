"""The code reaches the main network at every hidden layer (its own scale and shift there). Per layer: how many
code directions are injected, and how many of them reach the field and the surface?

usage: layer_inject.py <out.json> <label>=<checkpoint> [...]

For 6 random codes and each of the 63 directions on the sphere at the code, the code is moved by +/- 1e-3 and
ONLY layer k's scale and shift are taken from the moved code (all other layers keep the unmoved code's). The
differences give, for each layer k (63 columns each):
  injected:      the change of layer k's scale and shift themselves (1,024 values);
  at layer k:    the change of layer k's output (after the activation, at 1,024 points in the ball);
  at the field:  the change of the field at the same points;
  at the surface: the outward surface motion (field change / slope at about 800 surface points), pose removed.
Also the same with every layer's scale and shift moved together ("all layers").
For each block: effective number of directions ((sum s^2)^2 / sum s^4 over the 63 singular values), directions
above 1% of the largest, and size = squared Frobenius norm. "share" = this layer's size over the sum over layers.
"""
import json
import sys

import numpy as np
import torch

from arm_planner.data_free.decoder import Modulation
from arm_planner.data_free.latent import tangent_basis
from arm_planner.data_free.train import Trainer

STEP, CODES, POINTS, CHUNK = 1e-3, 6, 1024, 21


def measures(J):
    s2 = torch.linalg.eigvalsh(J.T @ J).clamp_min(0).flip(0).cpu().numpy()
    s = np.sqrt(s2)
    return {"effective": float(s2.sum() ** 2 / max((s2 ** 2).sum(), 1e-30)),
            "above1pct": int((s > 0.01 * s[0]).sum()) if s[0] > 0 else 0, "size": float(s2.sum())}


out = {}
for item in sys.argv[2:]:
    label, path = item.split("=", 1)
    trainer = Trainer.from_checkpoint(path, device="cuda", tensorboard=False)
    cfg, model = trainer.cfg, trainer.model.eval()
    L = len(model.hidden)
    gen = torch.Generator().manual_seed(909)
    codes = torch.nn.functional.normalize(torch.randn(CODES, int(cfg.latent_dim), generator=gen), dim=-1).cuda()
    x = torch.randn(POINTS, 3, generator=gen)
    x = (x / x.norm(dim=-1, keepdim=True) * 0.9 * torch.rand(POINTS, 1, generator=gen) ** (1 / 3)).cuda()
    acc = {}
    for c in range(CODES):
        z = codes[c]
        with torch.no_grad():
            f0 = model(z[None], x[None])[0]
        p = x[f0.abs().topk(800, largest=False).indices][None].clone()
        for _ in range(6):
            p = p.detach().requires_grad_(True)
            v = model(z[None], p)
            (g,) = torch.autograd.grad(v.sum(), p)
            p = (p - (v.detach()[..., None] * g / g.square().sum(-1, keepdim=True).clamp_min(1e-4)).clamp(-0.05, 0.05)).detach()
        p = p.requires_grad_(True)
        (g,) = torch.autograd.grad(model(z[None], p).sum(), p)
        p, g = p.detach()[0], g.detach()[0]
        n = -g / g.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        eye = torch.eye(3, device=p.device)
        B = torch.cat([n, torch.stack([(n * torch.linalg.cross(eye[i].expand_as(p), p)).sum(-1) for i in range(3)], 1),
                       (n * p).sum(-1, keepdim=True)], 1)
        pts = torch.cat([x, p])[None]
        T = tangent_basis(z)
        with torch.no_grad():
            base = model.prepare(z[None])
            moved = {}
            for sign in (1.0, -1.0):
                zz = torch.nn.functional.normalize(z[None] + sign * STEP * T, dim=-1)
                moved[sign] = model.prepare(zz)                                      # 63 x L x W
            for k in list(range(L)) + ["all"]:
                cols = {"injected": [], "at layer": [], "field": [], "surface": []}
                for s0 in range(0, T.shape[0], CHUNK):
                    m = T[s0:s0 + CHUNK].shape[0]
                    res = {}
                    for sign in (1.0, -1.0):
                        sc = base.scale.expand(m, -1, -1).clone()
                        sh = base.shift.expand(m, -1, -1).clone()
                        if k == "all":
                            sc, sh = moved[sign].scale[s0:s0 + m], moved[sign].shift[s0:s0 + m]
                        else:
                            sc[:, k], sh[:, k] = moved[sign].scale[s0:s0 + m, k], moved[sign].shift[s0:s0 + m, k]
                        mod = Modulation(sc, sh)
                        pre = []
                        model._net(mod, pts.expand(m, -1, -1), collect=pre)
                        layer_out = model.activation(pre[L - 1 if k == "all" else k])[:, :POINTS]
                        field = model.query(mod, pts.expand(m, -1, -1))
                        inj = torch.cat([sc[:, L - 1 if k == "all" else k], sh[:, L - 1 if k == "all" else k]], -1)
                        res[sign] = (inj, layer_out.reshape(m, -1), field[:, :POINTS], field[:, POINTS:])
                    for name, idx in (("injected", 0), ("at layer", 1), ("field", 2), ("surface", 3)):
                        cols[name].append(((res[1.0][idx] - res[-1.0][idx]) / (2 * STEP)).T)
                for name, blocks in cols.items():
                    J = torch.cat(blocks, dim=1)
                    if name == "surface":
                        J = J / g.norm(dim=-1).clamp_min(1e-6)[:, None]
                        J = J - B @ torch.linalg.lstsq(B, J).solution
                    acc.setdefault((str(k), name), []).append(measures(J))
    res = {"update": int(trainer.step_count)}
    for (k, name), vals in acc.items():
        res[f"{k}|{name}"] = {m: float(np.mean([v[m] for v in vals])) for m in vals[0]}
    out[label] = res
    tot = {name: sum(res[f"{k}|{name}"]["size"] for k in map(str, range(L))) for name in ("field", "surface")}
    print(f"\n{label} (update {res['update']}): code directions injected at each layer and how many reach the shape", flush=True)
    print(f"  {'':10s} {'scale & shift':>15s} {'layer output':>15s} {'field':>22s} {'surface (pose removed)':>26s}")
    print(f"  {'layer':10s} {'eff / >1%':>15s} {'eff / >1%':>15s} {'eff / >1% / share':>22s} {'eff / >1% / share':>26s}")
    for k in list(map(str, range(L))) + ["all"]:
        a, b, f, s_ = (res[f"{k}|{nm}"] for nm in ("injected", "at layer", "field", "surface"))
        name = f"main {int(k) + 1}" if k != "all" else "all layers"
        fs = f"{f['size'] / tot['field']:.2f}" if k != "all" else "-"
        ss = f"{s_['size'] / tot['surface']:.2f}" if k != "all" else "-"
        print(f"  {name:10s} {a['effective']:6.1f} / {a['above1pct']:4.0f}   {b['effective']:6.1f} / {b['above1pct']:4.0f}   "
              f"{f['effective']:6.1f} / {f['above1pct']:4.0f} / {fs:>5s}   {s_['effective']:8.1f} / {s_['above1pct']:4.0f} / {ss:>5s}", flush=True)
    del trainer, model
    torch.cuda.empty_cache()
json.dump(out, open(sys.argv[1], "w"), indent=1)
