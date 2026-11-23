"""CPU-only. How many of the graph's surface samples come from far source points?
A guess x' = x - f g is a surface sample (surfel) of the graph when |f(x')| <= delta = max(2 e, 0.01)."""
import sys
import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(6)
ROOT = "/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/"
run, ckpt = (sys.argv[1].split(":") + ["checkpoint_last.pt"])[:2]
from arm_planner.data_free.config import DataFreeConfig
from arm_planner.data_free.decoder import ModulatedSphereDecoder
from arm_planner.data_free.field import field_and_grad

state = torch.load(f"{ROOT}{run}/{ckpt}", map_location="cpu", weights_only=False)
cfg = DataFreeConfig.from_dict(state["config"])
model = ModulatedSphereDecoder(cfg)
model.load_state_dict(state["model"])
model.eval()
g = torch.Generator().manual_seed(99)
B, N = 24, 2048
z = F.normalize(torch.randn(B, cfg.latent_dim, generator=g), dim=-1)
x = (torch.rand(1, N, 3, generator=g) * 2 - 1).expand(B, -1, -1).contiguous()
mod = model.prepare(z)
f, gr = field_and_grad(model, mod, x, create_graph=False)
with torch.no_grad():
    xp = x - f[..., None] * gr
    fp = model.query(mod, xp)
f = f.detach()
e = float(fp.abs().median())
delta = max(2 * e, 0.01)
surf = fp.abs() <= delta
near = f > -0.15
near30 = f > -0.30
r_s = cfg.r_s()
print("run", run, ckpt, "| e (median |f(x')|, all points)", round(e, 4), "delta", round(delta, 4), "| graph surface edge radius r_s", round(r_s, 4), "min feature", cfg.min_feature)
print(f"surface samples per shape (of {N}): all sources {float(surf.float().sum(1).mean()):.0f} | sources with f > -0.15: {float((surf & near).float().sum(1).mean()):.0f} | f > -0.30: {float((surf & near30).float().sum(1).mean()):.0f}")
print(f"share of guesses that are surface samples: all {float(surf.float().mean()):.2f}; from f > -0.15 sources {float(surf[near].float().mean()):.2f}; from f < -0.30 sources {float(surf[~near30].float().mean()):.2f}")


def spacing(mask):
    out = []
    for b in range(B):
        p = xp[b][mask[b]]
        if p.shape[0] < 3:
            continue
        d = torch.cdist(p, p)
        d.fill_diagonal_(float("inf"))
        kth = d.topk(3, dim=1, largest=False).values       # 1st..3rd nearest
        out.append((float(kth[:, 0].mean()), float(kth[:, 2].mean()), float((d < r_s).float().sum(1).mean()), float(((d < r_s).sum(1) == 0).float().mean())))
    return np.array(out).mean(0)


for label, mask in (("all sources", surf), ("f > -0.15 only", surf & near), ("f > -0.30 only", surf & near30)):
    s = spacing(mask)
    print(f"  {label:16s} nearest surfel {s[0]:.3f}, 3rd nearest {s[1]:.3f}, neighbours within r_s {s[2]:.1f}, isolated surfels {100*s[3]:.1f}%")
# what ball sampling would give: box points -> ball of radius rb keeps the same density of near points times 8 / ball volume
for rb in (0.9, 0.95, 1.0):
    gain = 8.0 / (4 / 3 * np.pi * rb ** 3)
    print(f"  drawing the {N} points in a ball of radius {rb}: near-source surfels x{gain:.2f} -> f>-0.15: {gain*float((surf & near).float().sum(1).mean()):.0f}, f>-0.30: {gain*float((surf & near30).float().sum(1).mean()):.0f}; points beyond 0.9 left as outside anchors: {N*(1-(0.9/rb)**3) if rb>0.9 else 0:.0f}")
