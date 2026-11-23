"""CPU-only. Pose of random-code shapes from a dense sample of the ball (hard inside indicator), to see
whether pose error is a general offset or a few outlier codes; and what export-time normalisation would remove."""
import sys
import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(6)
ROOT = "/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/"
run, ckpt = (sys.argv[1].split(":") + ["checkpoint_last.pt"])[:2]
ncodes = int(sys.argv[2]); npts = int(sys.argv[3])
from arm_planner.data_free.config import DataFreeConfig
from arm_planner.data_free.decoder import ModulatedSphereDecoder

state = torch.load(f"{ROOT}{run}/{ckpt}", map_location="cpu", weights_only=False)
cfg = DataFreeConfig.from_dict(state["config"])
model = ModulatedSphereDecoder(cfg)
model.load_state_dict(state["model"])
model.eval()
g = torch.Generator().manual_seed(2024)
z = F.normalize(torch.randn(ncodes, cfg.latent_dim, generator=g), dim=-1)
x = torch.randn(npts, 3, generator=g)
x = x / x.norm(dim=-1, keepdim=True) * 0.9 * torch.rand(npts, 1, generator=g) ** (1 / 3)
with torch.no_grad():
    f = torch.cat([model(z[i:i + 16], x[None].expand(z[i:i + 16].shape[0], -1, -1)) for i in range(0, ncodes, 16)])
ins = (f > 0).float()
cnt = ins.sum(1)
c = (ins[..., None] * x[None]).sum(1) / cnt[:, None]
off = x[None] - c[:, None]
cov = torch.einsum("bn,bni,bnj->bij", ins, off, off) / cnt[:, None, None]
tr = torch.diagonal(cov, dim1=-2, dim2=-1).sum(-1)
d = torch.sqrt(torch.diagonal(cov, dim1=-2, dim2=-1))
corr = cov / (d[:, :, None] * d[:, None, :])
offd = torch.stack([corr[:, 0, 1], corr[:, 0, 2], corr[:, 1, 2]], 1).abs()
co = c.norm(dim=-1)
hard_target = 3 * cfg.base_radius ** 2 / 5
print("run", run, ckpt, "step", state["step"], "codes", ncodes, "points in ball", npts, "inside per code", int(cnt.mean()))
print(f"centroid offset: mean {float(co.mean()):.4f} median {float(co.median()):.4f} p90 {float(co.quantile(0.9)):.4f} max {float(co.max()):.4f}; mean of squares {float(co.square().mean()):.2e} (training term estimates this)")
print(f"mean centroid over codes (common offset): {np.array2string(c.mean(0).numpy(), precision=4)} |.|={float(c.mean(0).norm()):.4f}; rms about that mean {float((c - c.mean(0)).norm(dim=-1).square().mean().sqrt()):.4f}")
print(f"covariance trace (hard indicator): mean {float(tr.mean()):.4f} vs hard-ball target {hard_target:.4f}; spread {float(tr.std()/tr.mean()):.3f}; size = sqrt(trace): spread {float(tr.sqrt().std()/tr.sqrt().mean()):.3f}")
print(f"off-diagonal |correlation|: mean {float(offd.mean()):.3f} p90 {float(offd.flatten().quantile(0.9)):.3f} max {float(offd.max()):.3f}")
ev = torch.linalg.eigvalsh(cov)
print(f"axis ratios sqrt(l_max/l_min): median {float((ev[:,2]/ev[:,0]).sqrt().median()):.2f} p90 {float((ev[:,2]/ev[:,0]).sqrt().quantile(0.9)):.2f}")
vol = cnt / npts * (4 / 3 * np.pi * 0.9 ** 3)
print(f"volume: mean {float(vol.mean()):.3f} spread {float(vol.std()/vol.mean()):.3f}")
rmax = torch.stack([x[f[i] > 0].norm(dim=-1).max() for i in range(ncodes)])
print(f"largest radius reached by each shape: median {float(rmax.median()):.3f} p90 {float(rmax.quantile(0.9)):.3f} max {float(rmax.max()):.3f} (bound 0.85)")
