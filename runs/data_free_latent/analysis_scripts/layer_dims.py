"""How many independent directions of the code survive at each stage of the decoder?

usage: layer_dims.py <out.json> <label>=<checkpoint> [...]

For 8 random codes, the code is moved by +/- 1e-3 along each of the 63 directions of the sphere at that code,
and every stage's output is differenced (a numerical Jacobian, 63 columns). Stages: side network layers 1 and
2; the scale-and-shift output (all 8 x 1,024 values, and those of main layers 1 and 8 alone); each of the 8
hidden layers of the main network (after the activation, at 1,024 fixed points in the ball); the field at the
same points; the motion of the surface (field change / slope at about 800 surface points, with the 7
pose directions - shift, turn, scale - taken out).

For each stage and code, from the 63 singular values s of the Jacobian:
  effective dimension = (sum s^2)^2 / sum s^4 (1 if one direction carries everything, 63 if all are equal);
  dims for 90% = how many directions carry 90% of the sum of s^2;
  dims above 1% = how many s are over 1% of the largest.
Averaged over the 8 codes.
"""
import json
import sys

import numpy as np
import torch

from arm_planner.data_free.latent import tangent_basis
from arm_planner.data_free.train import Trainer

STEP, CODES, POINTS = 1e-3, 8, 1024


def measures(J):
    s2 = torch.linalg.eigvalsh(J.T @ J).clamp_min(0).flip(0).cpu().numpy()
    s = np.sqrt(s2)
    pr = s2.sum() ** 2 / max((s2 ** 2).sum(), 1e-30)
    cum = np.cumsum(s2) / max(s2.sum(), 1e-30)
    return {"effective": float(pr), "n90": int(np.searchsorted(cum, 0.9) + 1), "above1pct": int((s > 0.01 * s[0]).sum())}


out = {}
for item in sys.argv[2:]:
    label, path = item.split("=", 1)
    trainer = Trainer.from_checkpoint(path, device="cuda", tensorboard=False)
    cfg, model = trainer.cfg, trainer.model.eval()
    gen = torch.Generator().manual_seed(808)
    codes = torch.nn.functional.normalize(torch.randn(CODES, int(cfg.latent_dim), generator=gen), dim=-1).cuda()
    x = torch.randn(POINTS, 3, generator=gen)
    x = (x / x.norm(dim=-1, keepdim=True) * 0.9 * torch.rand(POINTS, 1, generator=gen) ** (1 / 3)).cuda()[None]
    L = len(model.hidden)

    def stages(z, p):
        z = z[None]
        out = {}
        h = torch.nn.functional.normalize(z, dim=-1)
        layer = 0
        for m in model.mapping:
            h = m(h)
            if isinstance(m, torch.nn.Softplus):
                layer += 1
                out[f"side network, layer {layer}"] = h[0]
        mod = model.prepare(z)
        film = torch.cat([mod.scale, mod.shift], dim=-1)[0]                    # L x 2W
        out["scale and shift, all layers"] = film.reshape(-1)
        out["scale and shift for main layer 1"] = film[0]
        out[f"scale and shift for main layer {L}"] = film[L - 1]
        pre = []
        model._net(mod, x, collect=pre)
        for k, v in enumerate(pre):
            out[f"main network, hidden layer {k + 1}"] = model.activation(v)[0].reshape(-1)
        out["field at points in the ball"] = model.query(mod, x)[0]
        out["surface points"] = model.query(mod, p[None])[0]
        return out

    rows = {}
    for c in range(CODES):
        z = codes[c]
        # surface points of this code's shape
        with torch.no_grad():
            f0 = model(z[None], x)[0]
        p = x[0][f0.abs().topk(800, largest=False).indices][None].clone()
        for _ in range(6):
            p = p.detach().requires_grad_(True)
            v = model(z[None], p)
            (g,) = torch.autograd.grad(v.sum(), p)
            p = (p - (v.detach()[..., None] * g / g.square().sum(-1, keepdim=True).clamp_min(1e-4)).clamp(-0.05, 0.05)).detach()
        p = p.requires_grad_(True)
        (g,) = torch.autograd.grad(model(z[None], p).sum(), p)
        p, g = p.detach()[0], g.detach()[0]
        T = tangent_basis(z)
        cols = {}
        with torch.no_grad():
            for j in range(T.shape[0]):
                plus = stages(torch.nn.functional.normalize(z + STEP * T[j], dim=-1), p)
                minus = stages(torch.nn.functional.normalize(z - STEP * T[j], dim=-1), p)
                for key in plus:
                    cols.setdefault(key, []).append(((plus[key] - minus[key]) / (2 * STEP)).float())
        for key, c_list in cols.items():
            J = torch.stack(c_list, dim=1)                                      # outputs x 63
            if key == "surface points":
                n = -g / g.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                J = J / g.norm(dim=-1).clamp_min(1e-6)[:, None]                 # outward surface motion per code step
                eye = torch.eye(3, device=J.device)
                B = torch.cat([n, torch.stack([(n * torch.linalg.cross(eye[i].expand_as(p), p)).sum(-1) for i in range(3)], 1),
                               (n * p).sum(-1, keepdim=True)], 1)
                J = J - B @ torch.linalg.lstsq(B, J).solution
                key = "surface motion (pose removed)"
            rows.setdefault(key, []).append(measures(J))
    res = {"update": int(trainer.step_count)}
    for key, vals in rows.items():
        res[key] = {m: float(np.mean([v[m] for v in vals])) for m in vals[0]}
    out[label] = res
    print(f"\n{label} (update {res['update']}): independent code directions at each stage (of 63)", flush=True)
    print(f"  {'stage':38s} {'effective':>10s} {'for 90%':>8s} {'above 1%':>9s}")
    for key, v in res.items():
        if key == "update":
            continue
        print(f"  {key:38s} {v['effective']:10.1f} {v['n90']:8.1f} {v['above1pct']:9.1f}", flush=True)
    del trainer, model
    torch.cuda.empty_cache()
json.dump(out, open(sys.argv[1], "w"), indent=1)
