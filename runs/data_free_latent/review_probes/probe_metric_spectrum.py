"""CPU-only. How do the two pair distances weight the latent directions the decoder already reads?
At a few codes, finite-difference the field along the 31 tangent directions at shared ball points and take the
spectrum of the field-change patterns under (a) all points with |x| < R_b, (b) only points with f > -d
(the clamped / collision-band distance; a clamped far point has zero derivative), (c) far points only.
Squared patterns are a proxy for the L1 distances used in training."""
import sys
import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(6)
ROOT = "/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/"
run, ckpt = (sys.argv[1].split(":") + ["checkpoint_last.pt"])[:2]
d = float(sys.argv[2]) if len(sys.argv) > 2 else 0.15
from arm_planner.data_free.config import DataFreeConfig
from arm_planner.data_free.decoder import ModulatedSphereDecoder
from arm_planner.data_free.latent import tangent_basis

state = torch.load(f"{ROOT}{run}/{ckpt}", map_location="cpu", weights_only=False)
cfg = DataFreeConfig.from_dict(state["config"])
model = ModulatedSphereDecoder(cfg)
model.load_state_dict(state["model"])
model.eval()
g = torch.Generator().manual_seed(5)
ncodes, npts, step = 10, 1500, 1e-2
x = torch.randn(npts, 3, generator=g)
x = x / x.norm(dim=-1, keepdim=True) * 0.9 * torch.rand(npts, 1, generator=g) ** (1 / 3)
z = F.normalize(torch.randn(ncodes, cfg.latent_dim, generator=g), dim=-1)


def summ(v):
    v = np.sort(np.maximum(v, 0))[::-1]
    p = v / v.sum()
    return float(np.exp(-(p * np.log(p + 1e-30)).sum())), float(p[0]), float(p[:3].sum()), float(v.sum())


acc = {k: [] for k in ("ball", "band", "far")}
with torch.no_grad():
    for k in range(ncodes):
        T = tangent_basis(z[k])
        zs = F.normalize(torch.cat([z[k] + step * T, z[k] - step * T]), dim=-1)
        f0 = model(z[k:k + 1], x[None])[0]
        vals = torch.cat([model(zs[i:i + 16], x[None].expand(zs[i:i + 16].shape[0], -1, -1)) for i in range(0, 62, 16)])
        D = ((vals[:31] - vals[31:]) / (2 * step)).T            # points x 31, field change per unit chord
        near = f0 > -d
        for name, M in (("ball", D), ("band", D[near]), ("far", D[~near])):
            s = torch.linalg.svdvals(M / np.sqrt(npts)).numpy() ** 2    # all divided by the same point count
            acc[name].append(s)
        if k == 0:
            print("share of ball points with f > -d:", float(near.float().mean()))
print("run", run, ckpt, "step", state["step"], "d", d, "codes", ncodes)
print(f"{'points used':28s} {'eff. directions':>15s} {'top-1 share':>12s} {'top-3 share':>12s} {'total':>10s}")
for name, label in (("ball", "all |x|<0.9 (as trained)"), ("band", f"f > -{d} only (clamped)"), ("far", f"f < -{d} only")):
    v = np.mean(acc[name], axis=0)
    e, p1, p3, tot = summ(v)
    print(f"{label:28s} {e:15.2f} {p1:12.3f} {p3:12.3f} {tot:10.4f}")
    print("   spectrum (first 12):", np.array2string(np.sort(v)[::-1][:12], precision=4))
