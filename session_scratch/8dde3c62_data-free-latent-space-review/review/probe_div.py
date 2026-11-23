"""CPU-only, round two. Force of the diversity loss alone on the ignored and the read subspace, many batches
(no second derivative needed: the pose fit uses detached gradients).

Two definitions of the pair distance:
  ball : as trained, mean |f_i - f_j| over the shared points with |x| < R_b (pose removed)
  tsdf : fields clamped below at -d before differencing, only points where either shape reads above -d,
         divided by the number of points in the ball (so far-far points contribute exactly zero)
Loss: -log(soft-min over the other shapes (power 4, over the k nearest if k given) / slope + 1e-3), mean over shapes.
"""
import sys, time
import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(6)
ROOT = "/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/"
spec = sys.argv[1]
run, ckpt = (spec.split(":") + ["checkpoint_last.pt"])[:2]
B = int(sys.argv[2]); N = int(sys.argv[3]); reps = int(sys.argv[4]); k_live = int(sys.argv[5])
knn = int(sys.argv[6]) if len(sys.argv) > 6 else 0
d_band = float(sys.argv[7]) if len(sys.argv) > 7 else 0.15

from arm_planner.data_free.config import DataFreeConfig
from arm_planner.data_free.decoder import ModulatedSphereDecoder, Modulation
from arm_planner.data_free import losses as L

state = torch.load(f"{ROOT}{run}/{ckpt}", map_location="cpu", weights_only=False)
cfg = DataFreeConfig.from_dict(state["config"])
model = ModulatedSphereDecoder(cfg)
model.load_state_dict(state["model"])
model.eval()
for p in model.parameters():
    p.requires_grad_(False)
g0 = torch.Generator().manual_seed(0)
zz = F.normalize(torch.randn(2048, cfg.latent_dim, generator=g0), dim=-1)
with torch.no_grad():
    film = model.film(model.mapping(zz))
sol = torch.linalg.lstsq(zz - zz.mean(0), film - film.mean(0)).solution
_, S, V = torch.linalg.svd(sol.T, full_matrices=False)
P = V[:k_live]
print("run", run, ckpt, "step", state["step"], "k_live", k_live, "codes", B, "points", N, "knn", knn, "d", d_band)
slope, power = float(cfg.diversity_slope), float(cfg.nearest_power)
first, second = torch.triu_indices(B, B, 1)


def pair_loss(f, g, x, clamp):
    bounded = (x[0].square().sum(-1) < float(cfg.bound_radius) ** 2)
    if clamp is None:
        mask = bounded[None].expand(first.numel(), -1).to(f.dtype)
        norm = mask.sum(1)
        field = f
    else:
        near = (torch.maximum(f[first], f[second]).detach() > -clamp) & bounded[None]
        mask = near.to(f.dtype)
        norm = bounded.to(f.dtype).sum().expand(first.numel())
        field = f.clamp(min=-clamp)
    delta = field[first] - field[second]
    delta = L._remove_pose(delta, x[first], 0.5 * (g[first] + g[second]), 0.5 * (f[first] + f[second]).detach(), mask)
    dist = (delta.abs() * mask).sum(1) / norm.clamp_min(1.0)
    dm = f.new_full((B, B), float("inf"))
    dm[first, second] = dist
    dm[second, first] = dist
    if knn > 0:
        dm = dm.topk(knn, dim=1, largest=False).values
    soft = (dm + 1e-6).pow(-power).sum(1).pow(-1.0 / power)
    return -torch.log(soft / slope + 1e-3).mean(), float(dm.min(1).values.mean())


def m(z):
    return model.film(model.mapping(z))


gen = torch.Generator().manual_seed(777)
res = {k: {"v": [], "a": [], "b": [], "nn": []} for k in ("ball", "tsdf")}
t0 = time.time()
for rep in range(reps):
    z = F.normalize(torch.randn(B, cfg.latent_dim, generator=gen), dim=-1)
    x = torch.rand(1, N, 3, generator=gen) * 2 - 1
    x[:, 0] = 0
    x = x.expand(B, -1, -1).contiguous().requires_grad_(True)
    zl = z @ P.T @ P
    zd = z - zl
    with torch.no_grad():
        base = m(z)
        odd_dead = 0.5 * (base - m(zl - zd))
        odd_live = 0.5 * (base - m(zd - zl))
    a = torch.tensor(1.0, requires_grad=True)
    b = torch.tensor(1.0, requires_grad=True)
    out = base + (a - 1.0) * odd_dead + (b - 1.0) * odd_live
    scale, shift = out.view(B, 2, cfg.layers, cfg.width).unbind(1)
    f = model.query(Modulation(scale, shift), x)
    (g,) = torch.autograd.grad(f.sum(), x, retain_graph=True)
    g = g.detach(); xd = x.detach()
    for name, clamp in (("ball", None), ("tsdf", d_band)):
        loss, nn = pair_loss(f, g, xd, clamp)
        ga, gb = torch.autograd.grad(loss, [a, b], retain_graph=True)
        res[name]["v"].append(float(loss)); res[name]["a"].append(float(ga)); res[name]["b"].append(float(gb)); res[name]["nn"].append(nn)
    if rep == 0:
        print(f"first batch {time.time()-t0:.1f}s")
print(f"{reps} batches, {time.time()-t0:.0f}s")
sem = lambda v: float(np.std(v, ddof=1) / np.sqrt(len(v)))
for name in ("ball", "tsdf"):
    r = res[name]
    print(f"{name:5s} loss {np.mean(r['v']):.4f}  nearest distance {np.mean(r['nn']):.4f}   IGNORED d/da {np.mean(r['a']):+.5f} +- {sem(r['a']):.5f}   READ d/db {np.mean(r['b']):+.4f} +- {sem(r['b']):.4f}")
