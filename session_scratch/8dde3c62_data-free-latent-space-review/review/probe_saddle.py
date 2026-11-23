"""CPU-only. Is the dependence on dead latent directions wanted or damped by each loss term?

z_hat = beta * z_live + alpha * z_dead (alpha = beta = 1 is the trained model).
Reports each term and d(term)/d(log alpha), d(term)/d(log beta), averaged over a few batches.
Negative derivative of the weighted total = gradient descent would GROW that dependence.
"""
import sys, time
import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(6)
ROOT = "/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/"
run = sys.argv[1]
B = int(sys.argv[2]); N = int(sys.argv[3]); reps = int(sys.argv[4])
alphas = [float(a) for a in sys.argv[5].split(",")]
k_live = int(sys.argv[6]) if len(sys.argv) > 6 else 6

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

# live subspace: top right-singular vectors of the linear z -> film map
g = torch.Generator().manual_seed(0)
zz = F.normalize(torch.randn(2048, cfg.latent_dim, generator=g), dim=-1)
with torch.no_grad():
    film = model.film(model.mapping(zz))
sol = torch.linalg.lstsq(zz - zz.mean(0), film - film.mean(0)).solution
_, S, V = torch.linalg.svd(sol.T, full_matrices=False)
P = V[:k_live]                       # k x 32
print("run", run, "step", state["step"], "k_live", k_live, "sing", np.array2string(S[:k_live + 3].numpy(), precision=2))
print("multipliers used:", {k: round(v, 1) for k, v in mult.items() if v != 1.0})

keys = ["diversity", "eikonal", "consistency", "barrier", "pose_centroid", "pose_scale", "pose_axes"]
gen = torch.Generator().manual_seed(123)
for alpha0 in alphas:
    acc = {k: [] for k in keys}
    da = {k: [] for k in keys}
    db = {k: [] for k in keys}
    t0 = time.time()
    for rep in range(reps):
        z = F.normalize(torch.randn(B, cfg.latent_dim, generator=gen), dim=-1)
        x = torch.rand(1, N, 3, generator=gen) * 2 - 1
        x[:, 0] = 0
        x = x.expand(B, -1, -1).contiguous()
        la = torch.tensor(float(np.log(alpha0)), requires_grad=True)
        lb = torch.tensor(0.0, requires_grad=True)
        z_live = z @ P.T @ P
        z_dead = z - z_live
        zh = torch.exp(lb) * z_live + torch.exp(la) * z_dead
        out = model.film(model.mapping(zh))
        scale, shift = out.view(B, 2, cfg.layers, cfg.width).unbind(1)
        mod = Modulation(scale, shift)
        f, gr = field_and_grad(model, mod, x, create_graph=True)
        xp = x - f[..., None] * gr
        fp = model.query(mod, xp)
        fb = FieldBatch(codes=z, cluster=torch.zeros(B, dtype=torch.long), x=x, f=f, g=gr, xp=xp, fp=fp, mod=mod)
        terms = {"eikonal": L.eikonal(fb), "consistency": L.consistency(fb), "barrier": L.barrier(fb, cfg)}
        terms.update(L.pose_terms(fb, cfg))
        div, info = L.diversity(fb, cfg, 0)
        terms["diversity"] = div
        for k in keys:
            ga, gb = torch.autograd.grad(terms[k], [la, lb], retain_graph=True, allow_unused=True)
            acc[k].append(float(terms[k])); da[k].append(float(ga)); db[k].append(float(gb))
    dt = time.time() - t0
    print(f"\nalpha={alpha0:g}  ({reps} batches of {B} codes x {N} points, {dt:.1f}s)   nearest_distance {float(info['nearest_distance']):.4f}")
    print(f"{'term':14s} {'value':>10s} {'mult':>7s} {'w*dT/dlog(alpha)':>20s} {'(sem)':>9s} {'w*dT/dlog(beta)':>18s} {'(sem)':>9s}")
    tot_a = np.zeros(reps); tot_b = np.zeros(reps)
    for k in keys:
        w = 1.0 if k == "diversity" else float(mult.get(k, 1.0))
        a = w * np.array(da[k]); b = w * np.array(db[k])
        tot_a += a; tot_b += b
        print(f"{k:14s} {np.mean(acc[k]):10.3e} {w:7.1f} {a.mean():20.4e} {a.std(ddof=1)/np.sqrt(reps) if reps>1 else 0:9.1e} {b.mean():18.4e} {b.std(ddof=1)/np.sqrt(reps) if reps>1 else 0:9.1e}")
    print(f"{'TOTAL':14s} {'':10s} {'':7s} {tot_a.mean():20.4e} {tot_a.std(ddof=1)/np.sqrt(reps) if reps>1 else 0:9.1e} {tot_b.mean():18.4e} {tot_b.std(ddof=1)/np.sqrt(reps) if reps>1 else 0:9.1e}")
    cons_a = tot_a - np.array(da["diversity"]); cons_b = tot_b - np.array(db["diversity"])
    print(f"{'constraints':14s} {'':10s} {'':7s} {cons_a.mean():20.4e} {'':9s} {cons_b.mean():18.4e}")
