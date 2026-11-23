"""CPU-only. Does the training pose estimator (split-sample, soft indicator, box points) read the same centroid
and scale error as a dense measurement of the same codes?"""
import sys
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(6)
ROOT = "/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/"
run, ckpt = (sys.argv[1].split(":") + ["checkpoint_last.pt"])[:2]
from arm_planner.data_free.config import DataFreeConfig
from arm_planner.data_free.decoder import ModulatedSphereDecoder
from arm_planner.data_free import losses as L

state = torch.load(f"{ROOT}{run}/{ckpt}", map_location="cpu", weights_only=False)
cfg = DataFreeConfig.from_dict(state["config"])
model = ModulatedSphereDecoder(cfg)
model.load_state_dict(state["model"])
model.eval()
g = torch.Generator().manual_seed(2024)
ncodes = 32
z = F.normalize(torch.randn(ncodes, cfg.latent_dim, generator=g), dim=-1)
# dense soft moments on box points (the trainer's definition, as in _probe_pose)
xd = torch.rand(40000, 3, generator=g) * 2 - 1
with torch.no_grad():
    fd = torch.cat([model(z[i:i + 8], xd[None].expand(8, -1, -1)) for i in range(0, ncodes, 8)])
    _, cen, cov = L.moments(SimpleNamespace(f=fd, x=xd[None].expand(ncodes, -1, -1)), cfg)
tr = torch.diagonal(cov, dim1=-2, dim2=-1).sum(-1)
target = float(cfg.cov_trace_target)
off = cov - torch.diag_embed(torch.diagonal(cov, dim1=-2, dim2=-1))
print("run", run, ckpt, "step", state["step"])
print(f"DENSE soft moments, {ncodes} codes x 40000 box points:")
print(f"  |centroid|^2 mean {float(cen.square().sum(-1).mean()):.2e}  (offset mean {float(cen.norm(dim=-1).mean()):.4f});  (trace-target)^2 mean {float((tr-target).square().mean()):.2e} (trace mean {float(tr.mean()):.4f}, target {target:.4f});  offdiag^2 sum mean {float(off.square().sum((-2,-1)).mean()):.2e}")
# the training estimator at N = 2048, many fresh point sets, same codes
vals = {k: [] for k in L.POSE_KEYS}
for rep in range(40):
    x = torch.rand(1, 2048, 3, generator=g) * 2 - 1
    x[:, 0] = 0
    x = x.expand(ncodes, -1, -1)
    with torch.no_grad():
        f = torch.cat([model(z[i:i + 16], x[i:i + 16]) for i in range(0, ncodes, 16)])
        out = L.pose_terms(SimpleNamespace(f=f, x=x), cfg)
    for k in L.POSE_KEYS:
        vals[k].append(float(out[k]))
print("TRAINING estimator (split halves, N=2048), mean over 40 point sets +- sem:")
for k in L.POSE_KEYS:
    v = np.array(vals[k])
    print(f"  {k:14s} {v.mean():.2e} +- {v.std(ddof=1)/np.sqrt(len(v)):.1e}   (per-set std {v.std(ddof=1):.1e}; tolerance {cfg.tolerances[k]:.1e})")
