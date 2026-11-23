"""CPU-only. Clean version: scale only the ODD (sign-flipping) part of the modulation's dependence
on the dead / live latent subspace, so no common-mode change of all shapes is mixed in.

mod(alpha, beta) = m(z) + (alpha-1) * odd_dead(z) + (beta-1) * odd_live(z)
odd_dead(z) = (m(zl + zd) - m(zl - zd)) / 2 ; odd_live(z) = (m(zl + zd) - m(-zl + zd)) / 2
alpha = beta = 1 is exactly the trained model. Reports w * d(term)/d(alpha) and d/d(beta).
Positive total => gradient descent shrinks that dependence.
"""
import sys, time
import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(6)
ROOT = "/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/"
run = sys.argv[1]
B = int(sys.argv[2]); N = int(sys.argv[3]); reps = int(sys.argv[4])
k_live = int(sys.argv[5]) if len(sys.argv) > 5 else 6

from arm_planner.data_free.config import DataFreeConfig
from arm_planner.data_free.decoder import ModulatedSphereDecoder, Modulation
from arm_planner.data_free.field import FieldBatch, field_and_grad
from arm_planner.data_free import losses as L

state = torch.load(f"{ROOT}{run}/checkpoint_last.pt", map_location="cpu", weights_only=False)
cfg = DataFreeConfig.from_dict(state["config"])
cfg.points = N
model = ModulatedSphereDecoder(cfg)
model.load_state_dict(state["model"])
model.eval()
for p in model.parameters():
    p.requires_grad_(False)
mult = state["multipliers"]["multipliers"]

g = torch.Generator().manual_seed(0)
zz = F.normalize(torch.randn(2048, cfg.latent_dim, generator=g), dim=-1)
with torch.no_grad():
    film = model.film(model.mapping(zz))
sol = torch.linalg.lstsq(zz - zz.mean(0), film - film.mean(0)).solution
_, S, V = torch.linalg.svd(sol.T, full_matrices=False)
P = V[:k_live]
print("run", run, "step", state["step"], "k_live", k_live, "mode", cfg.diversity_mode, "spread_weight", cfg.spread_weight, "global", cfg.global_codes)
print("multipliers used:", {k: round(v, 1) for k, v in mult.items() if v != 1.0})


def m(z):
    return model.film(model.mapping(z))


keys = ["diversity", "eikonal", "consistency", "barrier", "pose_centroid", "pose_scale", "pose_axes"]
extra = ["cons_far(|x|>=Rb)", "cons_mid(f<-.15)", "cons_near(f>-.15)", "spread(xw)"]
keys = keys + extra
gen = torch.Generator().manual_seed(123)
acc = {k: [] for k in keys}; da = {k: [] for k in keys}; db = {k: [] for k in keys}
t0 = time.time()
for rep in range(reps):
    if bool(cfg.global_codes):
        z = F.normalize(torch.randn(B, cfg.latent_dim, generator=gen), dim=-1)
        cluster_ids = torch.zeros(B, dtype=torch.long)
        x = torch.rand(1, N, 3, generator=gen) * 2 - 1
        x[:, 0] = 0
        x = x.expand(B, -1, -1).contiguous()
    else:
        # the trainer's cluster sampling (fresh centres, log-uniform chord radius), points shared per cluster
        import math
        from arm_planner.data_free.latent import sample_cluster, sample_sphere
        per = int(cfg.codes_per_cluster); ncl = B // per
        centres = sample_sphere(ncl, cfg.latent_dim, gen)
        lo, hi = math.log(float(cfg.eps_min)), math.log(float(cfg.eps_max))
        eps = torch.exp(lo + (hi - lo) * torch.rand(ncl, generator=gen))
        z = sample_cluster(centres, per, eps, gen).reshape(ncl * per, cfg.latent_dim)
        cluster_ids = torch.arange(ncl).repeat_interleave(per)
        xc = torch.rand(ncl, N, 3, generator=gen) * 2 - 1
        xc[:, 0] = 0
        x = xc[cluster_ids].contiguous()
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
    mod = Modulation(scale, shift)
    f, gr = field_and_grad(model, mod, x, create_graph=True)
    xp = x - f[..., None] * gr
    fp = model.query(mod, xp)
    fb = FieldBatch(codes=z, cluster=cluster_ids, x=x, f=f, g=gr, xp=xp, fp=fp, mod=mod)
    terms = {"eikonal": L.eikonal(fb), "consistency": L.consistency(fb), "barrier": L.barrier(fb, cfg)}
    terms.update(L.pose_terms(fb, cfg))
    div, info = L.diversity(fb, cfg, 0)
    if float(cfg.spread_weight) > 0:
        sp, _ = L.spread(fb, cfg)
        terms["spread(xw)"] = float(cfg.spread_weight) * sp
        div = div + float(cfg.spread_weight) * sp
    else:
        terms["spread(xw)"] = f.sum() * 0.0
    terms["diversity"] = div
    # consistency split by region (the three parts add up to the consistency term)
    absfp = fp[:, 1:].abs(); cnt = float(absfp.numel())
    rho_ = x[:, 1:].norm(dim=-1); fd = f[:, 1:].detach()
    far = rho_ >= cfg.bound_radius
    mid = (~far) & (fd < -0.15)
    near = (~far) & (fd >= -0.15)
    terms["cons_far(|x|>=Rb)"] = (absfp * far).sum() / cnt
    terms["cons_mid(f<-.15)"] = (absfp * mid).sum() / cnt
    terms["cons_near(f>-.15)"] = (absfp * near).sum() / cnt
    for k in keys:
        ga, gb = torch.autograd.grad(terms[k], [a, b], retain_graph=True, allow_unused=True)
        acc[k].append(float(terms[k])); da[k].append(float(ga)); db[k].append(float(gb))
    if rep == 0:
        print("relative size of odd parts: |odd_dead|/|odd_live| =", float(odd_dead.norm() / odd_live.norm()))
dt = time.time() - t0
print(f"\n{reps} batches of {B} codes x {N} points, {dt:.1f}s")
print(f"{'term':14s} {'value':>10s} {'mult':>7s} {'w*dT/d(alpha) DEAD':>20s} {'(sem)':>9s} {'w*dT/d(beta) LIVE':>18s} {'(sem)':>9s}")
tot_a = np.zeros(reps); tot_b = np.zeros(reps)
sem = lambda v: v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
for k in keys:
    w = 1.0 if k == "diversity" else float(mult.get(k, 1.0))
    if k in extra:
        w = 1.0 if k.startswith("spread") else float(mult.get("consistency", 1.0))
    va = w * np.array(da[k]); vb = w * np.array(db[k])
    if k not in extra:
        tot_a += va; tot_b += vb
    print(f"{k:14s} {np.mean(acc[k]):10.3e} {w:7.1f} {va.mean():20.4e} {sem(va):9.1e} {vb.mean():18.4e} {sem(vb):9.1e}")
print(f"{'TOTAL':14s} {'':10s} {'':7s} {tot_a.mean():20.4e} {sem(tot_a):9.1e} {tot_b.mean():18.4e} {sem(tot_b):9.1e}")
ca = tot_a - np.array(da["diversity"]); cb = tot_b - np.array(db["diversity"])
print(f"{'constraints':14s} {'':10s} {'':7s} {ca.mean():20.4e} {sem(ca):9.1e} {cb.mean():18.4e} {sem(cb):9.1e}")
