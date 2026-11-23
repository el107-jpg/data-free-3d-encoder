"""CPU-only field-level probes on a checkpoint: odd symmetry, where the SDF error sits, sampling waste, tanh bound."""
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
print("run", run, "step", state["step"])
g = torch.Generator().manual_seed(5)
B, N = 48, 4096
z = F.normalize(torch.randn(B, cfg.latent_dim, generator=g), dim=-1)
x = (torch.rand(1, N, 3, generator=g) * 2 - 1).expand(B, -1, -1).contiguous()
rho = x[0].norm(dim=-1)

# ---- (a) odd symmetry in z
with torch.no_grad():
    fz = model(z, x)
    fm = model(-z, x)
    zb = F.normalize(torch.randn(512, cfg.latent_dim, generator=g), dim=-1)
    fbar = torch.cat([model(zb[i:i + 64], x[:1, :1024].expand(64, -1, -1)) for i in range(0, 512, 64)]).mean(0)   # mean field at first 1024 pts
in_b = rho[:1024] < cfg.bound_radius
dz = (fz[:, :1024] - fbar)[:, in_b]
dm = (fm[:, :1024] - fbar)[:, in_b]
odd = 0.5 * (dz - dm); even = 0.5 * (dz + dm)
print(f"(a) corr(f(z)-mean, f(-z)-mean) over codes x points (|x|<R_b): {float((dz*dm).sum() / (dz.norm()*dm.norm())):.3f}")
print(f"    energy share: odd part {float(odd.square().sum() / dz.square().sum()):.3f}, even part {float(even.square().sum() / dz.square().sum()):.3f}")
# sign agreement: inside(z) vs outside(-z) relative to the mean shape
print(f"    mean field: inside fraction of mean shape {float((fbar[in_b] > 0).float().mean()):.3f}")

# ---- (b) where does the SDF error sit
mod = model.prepare(z)
f, gr = field_and_grad(model, mod, x, create_graph=False)
with torch.no_grad():
    xp = x - f[..., None] * gr
    fp = model.query(mod, xp).abs()
    gn = gr.norm(dim=-1)
f = f.detach()
eik = (gn - 1).square()
print(f"(b) all points: mean|fp| {float(fp.mean()):.4f} (tolerance {cfg.tolerances['consistency']}), median {float(fp.median()):.4f}, eikonal {float(eik.mean()):.5f}")
bins = [(-9, -0.6), (-0.6, -0.3), (-0.3, -0.15), (-0.15, -0.04), (-0.04, 0.04), (0.04, 0.15), (0.15, 9)]
tot = float(fp.sum())
print("    by signed field value f (negative = outside):  share of points | mean|fp| | share of total consistency | mean eikonal err")
for lo, hi in bins:
    msk = (f >= lo) & (f < hi)
    if msk.any():
        print(f"      f in [{lo:5.2f},{hi:5.2f}): {float(msk.float().mean()):6.3f} | {float(fp[msk].mean()):.4f} | {float(fp[msk].sum()) / tot:6.3f} | {float(eik[msk].mean()):.5f}")
far = (rho >= cfg.bound_radius)[None].expand(B, -1)
print(f"    points with |x| >= R_b={cfg.bound_radius}: share {float(far.float().mean()):.3f}, mean|fp| {float(fp[far].mean()):.4f}, share of total consistency {float(fp[far].sum()) / tot:.3f}")
print(f"    points with |x| <  R_b: mean|fp| {float(fp[~far].mean()):.4f}")

# ---- (c) sampling: how many of 2048 points are useful
inside = (f > 0).float().sum(1) * 2048 / N
band = (f.abs() < 0.5 * cfg.min_feature).float().sum(1) * 2048 / N
print(f"(c) per shape at N=2048: inside points mean {float(inside.mean()):.0f} (min {float(inside.min()):.0f}), within t/2 of the surface mean {float(band.mean()):.0f} (min {float(band.min()):.0f})")
print(f"    volume fraction of box: mean {float((f>0).float().mean()):.4f}; std across codes {float((f>0).float().mean(1).std()):.4f}")

# ---- (d) tanh bound usage
with torch.no_grad():
    corr = f - (cfg.base_radius - x.norm(dim=-1))
    surf = f.abs() < 0.02
    a = cfg.correction_bound
    c_s = corr[surf].abs() / a
    print(f"(d) |correction|/A at the surface: mean {float(c_s.mean()):.3f}, p90 {float(c_s.quantile(0.9)):.3f}, p99 {float(c_s.quantile(0.99)):.3f}, max {float(c_s.max()):.3f}; "
          f"tanh' there (gradient attenuation) p10 {float((1 - c_s.square()).quantile(0.1)):.3f}, min {float((1 - c_s.square()).min()):.3f}")
    rs = x.norm(dim=-1)[surf]
    print(f"    surface radius: min {float(rs.min()):.3f}, p1 {float(rs.quantile(0.01)):.3f}, median {float(rs.median()):.3f}, p99 {float(rs.quantile(0.99)):.3f}, max {float(rs.max()):.3f}  (allowed [{cfg.base_radius-a:.2f}, {cfg.base_radius+a:.2f}])")
    call = corr.abs() / a
    print(f"    |correction|/A over all points: mean {float(call.mean()):.3f}, share > 0.9: {float((call > 0.9).float().mean()):.3f}")

# ---- (e) convexity proxy: fraction of surface where mean curvature sign indicates concavity is costly to get on CPU; use star-shapedness instead
with torch.no_grad():
    radial = -(gr * x).sum(-1) / (gn * x.norm(dim=-1)).clamp_min(1e-9)      # cos between outward normal and radial direction
    print(f"(e) at the surface, cos(outward normal, radial): mean {float(radial[surf].mean()):.3f}, p5 {float(radial[surf].quantile(0.05)):.3f}, share < 0 (surface facing the centre, i.e. not star-shaped): {float((radial[surf] < 0).float().mean()):.4f}")
