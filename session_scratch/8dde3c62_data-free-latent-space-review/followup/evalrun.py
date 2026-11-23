"""One-stop measurement of a saved data-free model (any latent dimension).

usage: evalrun.py <checkpoint> <tag> [chunk]
Writes <tag>.json next to this script:
 - spectrum: RMS outward surface motion per unit chord along each tangent direction of the code,
   sorted at each of 24 fresh random codes and averaged (pose removed); total variance, effective
   number of directions, counts above thresholds;
 - pose of 64 fresh random codes (soft centroid offset, soft covariance trace), as the trainer measures it;
 - topology of 128 fresh random codes at resolution 96, flagged ones recounted at 128 and 160;
 - overlap with the anchor objects at their trained codes (models with anchors).
"""
import json
import os
import sys

import numpy as np
import torch

from arm_planner.data_free import diagnostics as D
from arm_planner.data_free.field import field_and_grad
from arm_planner.data_free.latent import tangent_basis
from arm_planner.data_free.train import Trainer

ckpt, tag = sys.argv[1], sys.argv[2]
chunk = int(sys.argv[3]) if len(sys.argv) > 3 else 8
HERE = os.path.dirname(os.path.abspath(__file__))
trainer = Trainer.from_checkpoint(ckpt, device="cuda", tensorboard=False)
cfg, model = trainer.cfg, trainer.model.eval()
dim = int(cfg.latent_dim)
out = {"checkpoint": ckpt, "update": trainer.step_count, "latent_dim": dim}

# ------------------------------------------------------------------ spectrum
gen = torch.Generator().manual_seed(7)
step, n_codes = 1e-3, 24
codes = torch.nn.functional.normalize(torch.randn(n_codes, dim, generator=gen), dim=-1).cuda()
local = []
for k in range(n_codes):
    z = codes[k]
    x = (torch.rand(1, 5000, 3, generator=gen) * 2 - 1).cuda()
    mod = model.prepare(z[None])
    f, g = field_and_grad(model, mod, x, create_graph=False)
    xp = (x - f[..., None] * g).detach()
    fp, gp = field_and_grad(model, mod, xp, create_graph=False)
    keep = (fp[0].abs() < 3 * fp[0].abs().median().clamp_min(1e-4)) & (gp[0].norm(dim=-1) > 0.3)
    p, gn = xp[0][keep].detach(), gp[0][keep].detach()
    n_out = -gn / gn.norm(dim=-1, keepdim=True)
    T = tangent_basis(z)
    zs = torch.cat([z + step * T, z - step * T])
    zs = zs / zs.norm(dim=-1, keepdim=True)
    with torch.no_grad():
        vals = torch.cat([model(zs[i:i + chunk], p[None].expand(zs[i:i + chunk].shape[0], -1, -1))
                          for i in range(0, zs.shape[0], chunk)])
    half = dim - 1
    Dm = ((vals[:half] - vals[half:]) / (2 * step) / gn.norm(dim=-1)[None]).T
    eye = torch.eye(3, device="cuda")
    B = torch.cat([n_out @ eye,
                   torch.stack([(n_out * torch.linalg.cross(eye[i].expand_as(p), p)).sum(-1) for i in range(3)], 1),
                   (n_out * p).sum(-1, keepdim=True)], 1)
    R = Dm - B @ torch.linalg.lstsq(B, Dm).solution
    s = torch.linalg.svdvals(R / np.sqrt(p.shape[0]))
    local.append((s ** 2).cpu().numpy())
    del f, g, fp, gp
var = np.array(local).mean(0)
share = var / var.sum()
rms = np.sqrt(var)
out["spectrum"] = {"rms": [float(v) for v in rms], "total_variance": float(var.sum()),
                   "effective_directions": float(np.exp(-(share * np.log(share + 1e-30)).sum())),
                   "above_0.02": int((rms > 0.02).sum()), "above_0.01": int((rms > 0.01).sum()),
                   "above_0.005": int((rms > 0.005).sum())}
print(tag, "spectrum:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out["spectrum"].items() if k != "rms"},
      "| top 16:", " ".join(f"{v:.3f}" for v in rms[:16]), flush=True)

# ------------------------------------------------------------------ pose of fresh codes
gen = torch.Generator().manual_seed(99)
fresh = torch.nn.functional.normalize(torch.randn(64, dim, generator=gen), dim=-1).cuda()
xs = (torch.rand(100000, 3, generator=gen) * 2 - 1).cuda()
cents, traces = [], []
with torch.no_grad():
    for i in range(64):
        f = torch.cat([model(fresh[i:i + 1], xs[s:s + 50000][None])[0] for s in range(0, xs.shape[0], 50000)])
        w = torch.sigmoid(f / float(cfg.soft_tau))
        c = (w[:, None] * xs).sum(0) / (w.sum() + 1e-6)
        off = xs - c
        traces.append(float((w * off.square().sum(-1)).sum() / (w.sum() + 1e-6)))
        cents.append(c)
cents = torch.stack(cents)
out["pose"] = {"centroid_offset_mean": float(cents.norm(dim=-1).mean()), "centroid_offset_max": float(cents.norm(dim=-1).max()),
               "common_shift": float(cents.mean(0).norm()), "trace_mean": float(np.mean(traces)),
               "trace_target": float(cfg.cov_trace_target), "trace_rel_std": float(np.std(traces) / np.mean(traces))}
print(tag, "pose:", {k: round(v, 4) for k, v in out["pose"].items()}, flush=True)

# ------------------------------------------------------------------ topology of fresh codes
gen = torch.Generator().manual_seed(12345)
count, res = 128, 96
tc = torch.nn.functional.normalize(torch.randn(count, dim, generator=gen), dim=-1).cuda()
rows = []
with torch.no_grad():
    for c0 in range(0, count, chunk):
        rows += D.decoder_grid_topology(model, tc[c0:c0 + chunk], cfg, resolution=res)
flag = [i for i, r in enumerate(rows) if r["pieces"] != 1 or r["cavities"] > 0 or r["handles"] > 0]
confirmed, detail = 0, []
for i in flag:
    bad = True
    line = {"code": i, "at_96": rows[i]}
    for r2 in (128, 160):
        with torch.no_grad():
            t = D.grid_topology(lambda p, i=i: model(tc[i:i + 1], p[None])[0], resolution=r2, device="cuda")
        line[f"at_{r2}"] = {k: int(t[k]) for k in ("pieces", "cavities", "handles")}
        bad = bad and (t["pieces"] != 1 or t["cavities"] > 0 or t["handles"] > 0)
    confirmed += int(bad)
    detail.append(line)
out["topology"] = {"codes": count, "clean_at_96": count - len(flag), "flagged": len(flag),
                   "defective_at_128_and_160": confirmed, "clean_on_fine_grids": count - confirmed,
                   "detail": [{k: (v if isinstance(v, (int, dict)) else int(v)) for k, v in d.items()} for d in detail]}
print(tag, "topology:", {k: v for k, v in out["topology"].items() if k != "detail"}, flush=True)

# ------------------------------------------------------------------ anchors
if trainer.anchor_codes is not None:
    a = trainer._probe_anchors(torch.Generator().manual_seed(5))
    out["anchors"] = {k.split("/")[-1]: float(v) for k, v in a.items()}
    print(tag, "anchors:", {k: round(v, 3) for k, v in out["anchors"].items()}, flush=True)


def plain(o):
    if isinstance(o, dict):
        return {k: plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [plain(v) for v in o]
    if hasattr(o, "item"):
        return o.item()
    return o


json.dump(plain(out), open(os.path.join(HERE, tag + ".json"), "w"), indent=1)
print("saved", os.path.join(HERE, tag + ".json"))
