"""CPU-only, round two. Consistency split by signed field value, for the read and the ignored subspace.

Same construction as probe_saddle2: scale only the odd (sign-flipping) part of the modulation's dependence
on the ignored / read latent subspace; alpha = beta = 1 is the trained model.
For each bin of f (detached) at the source point: share of points, mean |f(x')| in the bin, the bin's
contribution to the all-points consistency term, and multiplier * d(contribution)/d(alpha|beta).
Also prints the consistency derivative under three definitions (A all points, B band only, C none) and the
medians that would set the graph's band width.
"""
import sys, time
import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(6)
ROOT = "/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/"
run, ckpt = (sys.argv[1].split(":") + ["checkpoint_last.pt"])[:2]
B = int(sys.argv[2]); N = int(sys.argv[3]); reps = int(sys.argv[4]); k_live = int(sys.argv[5])

from arm_planner.data_free.config import DataFreeConfig
from arm_planner.data_free.decoder import ModulatedSphereDecoder, Modulation
from arm_planner.data_free.field import FieldBatch, field_and_grad
from arm_planner.data_free import losses as L

state = torch.load(f"{ROOT}{run}/{ckpt}", map_location="cpu", weights_only=False)
cfg = DataFreeConfig.from_dict(state["config"])
cfg.points = N
model = ModulatedSphereDecoder(cfg)
model.load_state_dict(state["model"])
model.eval()
for p in model.parameters():
    p.requires_grad_(False)
mult = state["multipliers"]["multipliers"]
lam = float(mult["consistency"])
g0 = torch.Generator().manual_seed(0)
zz = F.normalize(torch.randn(2048, cfg.latent_dim, generator=g0), dim=-1)
with torch.no_grad():
    film = model.film(model.mapping(zz))
sol = torch.linalg.lstsq(zz - zz.mean(0), film - film.mean(0)).solution
_, S, V = torch.linalg.svd(sol.T, full_matrices=False)
P = V[:k_live]
print("run", run, "step", state["step"], "k_live", k_live, "sing", np.array2string(S[:k_live + 3].numpy(), precision=1), "consistency multiplier", round(lam, 1))


def m(z):
    return model.film(model.mapping(z))


edges = [-9.0, -0.6, -0.3, -0.15, -0.04, 0.04, 0.15, 9.0]
names = [f"f in [{a:5.2f},{b:5.2f})" for a, b in zip(edges[:-1], edges[1:])]
gen = torch.Generator().manual_seed(321)
val = {n: [] for n in names}; share = {n: [] for n in names}; da = {n: [] for n in names}; db = {n: [] for n in names}
med_all, med_band, med_near, eik_d, bar_d = [], [], [], [], []
pose_a, pose_b, eik_a, eik_b = [], [], [], []
t0 = time.time()
for rep in range(reps):
    z = F.normalize(torch.randn(B, cfg.latent_dim, generator=gen), dim=-1)
    x = torch.rand(1, N, 3, generator=gen) * 2 - 1
    x[:, 0] = 0
    x = x.expand(B, -1, -1).contiguous()
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
    absfp = fp[:, 1:].abs(); cnt = float(absfp.numel())
    fd = f[:, 1:].detach()
    for n, lo, hi in zip(names, edges[:-1], edges[1:]):
        msk = (fd >= lo) & (fd < hi)
        part = (absfp * msk).sum() / cnt
        ga, gb = torch.autograd.grad(part, [a, b], retain_graph=True)
        val[n].append(float(part)); share[n].append(float(msk.float().mean()))
        da[n].append(lam * float(ga)); db[n].append(lam * float(gb))
    fb = FieldBatch(codes=z, cluster=torch.zeros(B, dtype=torch.long), x=x, f=f, g=gr, xp=xp, fp=fp, mod=mod)
    pose = L.pose_terms(fb, cfg)
    ptot = sum(float(mult.get(k, 1.0)) * v for k, v in pose.items())
    ga, gb = torch.autograd.grad(ptot, [a, b], retain_graph=True)
    pose_a.append(float(ga)); pose_b.append(float(gb))
    eik = float(mult.get("eikonal", 1.0)) * L.eikonal(fb)
    ga, gb = torch.autograd.grad(eik, [a, b], retain_graph=True)
    eik_a.append(float(ga)); eik_b.append(float(gb))
    with torch.no_grad():
        a_ = absfp.detach()
        med_all.append(float(a_.median()))
        med_band.append(float(a_[(fd > -0.15) & (fd < 0.04)].median()))
        med_near.append(float(a_[fd > -0.15].median()))
print(f"{reps} batches of {B} codes x {N} points, {time.time()-t0:.0f}s")
sem = lambda v: float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
print(f"{'bin':22s} {'pts share':>9s} {'mean|fp|':>9s} {'term part':>10s} {'IGNORED: mult*d/da':>20s} {'sem':>8s} {'READ: mult*d/db':>17s} {'sem':>8s} {'elasticity(read)':>16s}")
for n in names:
    v = np.mean(val[n]); s = np.mean(share[n])
    print(f"{n:22s} {s:9.3f} {v/max(s,1e-12):9.4f} {v:10.2e} {np.mean(da[n]):20.4e} {sem(da[n]):8.1e} {np.mean(db[n]):17.4e} {sem(db[n]):8.1e} {np.mean(db[n])/(lam*v):16.2f}")


def combo(sel):
    A = np.sum([np.array(da[n]) for n in sel], axis=0); Bv = np.sum([np.array(db[n]) for n in sel], axis=0)
    v = sum(np.mean(val[n]) for n in sel); s = sum(np.mean(share[n]) for n in sel)
    return A, Bv, v, s


print("\nConsistency under three definitions (same multiplier, each part still divided by ALL points):")
for label, sel in (("A all points", names), ("B band -0.15<f<0.04", names[3:5]), ("B' f>-0.15 (all inside)", names[3:]), ("B'' f>-0.30", names[2:])):
    A, Bv, v, s = combo(sel)
    print(f"  {label:26s} pts {s:5.3f}  mean|fp| over the set {v/s:.4f}   IGNORED {A.mean():+.4e} +- {sem(A):.1e}   READ {Bv.mean():+.4e} +- {sem(Bv):.1e}   read-force per unit multiplier if averaged over the set: {Bv.mean()/lam/s:.2e}")
print(f"  pose terms (weighted)        IGNORED {np.mean(pose_a):+.4e} +- {sem(pose_a):.1e}   READ {np.mean(pose_b):+.4e} +- {sem(pose_b):.1e}")
print(f"  eikonal (weighted)           IGNORED {np.mean(eik_a):+.4e} +- {sem(eik_a):.1e}   READ {np.mean(eik_b):+.4e} +- {sem(eik_b):.1e}")
print(f"median |f(x')|: all points {np.mean(med_all):.4f} | band -0.15<f<0.04 {np.mean(med_band):.4f} | f>-0.15 {np.mean(med_near):.4f}")
