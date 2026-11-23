"""CPU-only: where does the low rank live? mapping network vs trunk."""
import sys
import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(4)
ROOT = "/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/"
run = sys.argv[1]
state = torch.load(f"{ROOT}{run}/checkpoint_last.pt", map_location="cpu", weights_only=False)
sd = state["model"]
print("run", run, "step", state["step"])
from arm_planner.data_free.config import DataFreeConfig
from arm_planner.data_free.decoder import ModulatedSphereDecoder
cfg = DataFreeConfig.from_dict(state["config"])
model = ModulatedSphereDecoder(cfg)
model.load_state_dict(sd)
model.eval()


def eff(v):
    v = np.maximum(np.asarray(v, dtype=np.float64), 0)
    p = v / v.sum()
    return float(np.exp(-(p * np.log(p + 1e-300)).sum()))


def show(name, s):
    s = np.asarray(s)
    print(f"{name}: top12 {np.array2string(s[:12], precision=3)}  ... s[20]={s[min(20,len(s)-1)]:.3g} s[30]={s[min(30,len(s)-1)]:.3g}  eff(s^2)={eff(s**2):.2f}")


# 1. first mapping layer
W1 = sd["mapping.0.weight"]
show("svd mapping.0.weight (256x32)", torch.linalg.svdvals(W1).numpy())
fresh = ModulatedSphereDecoder(cfg)
show("  same at fresh init", torch.linalg.svdvals(fresh.mapping[0].weight.detach()).numpy())
print("  |W1 - init-scale| : trained fro", float(W1.norm()), "fresh fro", float(fresh.mapping[0].weight.norm()))

# 2. PCA of activations over random codes
g = torch.Generator().manual_seed(0)
z = F.normalize(torch.randn(2048, cfg.latent_dim, generator=g), dim=-1)
with torch.no_grad():
    h = z
    acts = []
    for layer in model.mapping:
        h = layer(h)
        acts.append(h)
    film = model.film(h)
    fh = z
    facts = []
    for layer in fresh.mapping:
        fh = layer(fh)
        facts.append(fh)
    ffilm = fresh.film(fh)
for name, a, fa in (("mapping layer1 post-act", acts[1], facts[1]), ("mapping layer2 post-act", acts[3], facts[3]), ("film output (8192)", film, ffilm)):
    s = torch.linalg.svdvals(a - a.mean(0, keepdim=True)).numpy() / np.sqrt(a.shape[0])
    fs = torch.linalg.svdvals(fa - fa.mean(0, keepdim=True)).numpy() / np.sqrt(a.shape[0])
    show(f"PCA sing. values over 2048 codes, {name}", s)
    show("   fresh init", fs)
scale, shift = film.view(-1, 2, cfg.layers, cfg.width).unbind(1)
print("film scale std across codes per layer:", [round(float(scale[:, l].std(0).mean()), 4) for l in range(cfg.layers)])
print("film shift std across codes per layer:", [round(float(shift[:, l].std(0).mean()), 4) for l in range(cfg.layers)])
print("film scale mean abs per layer:", [round(float(scale[:, l].mean(0).abs().mean()), 4) for l in range(cfg.layers)])
print("film shift mean abs per layer:", [round(float(shift[:, l].mean(0).abs().mean()), 4) for l in range(cfg.layers)])

# 3. regress film output on z linearly: how linear is the mapping, and rank of the linear map
zc = z - z.mean(0)
fc = film - film.mean(0)
sol = torch.linalg.lstsq(zc, fc).solution          # 32 x 8192
res = fc - zc @ sol
print("share of film variance explained by a linear function of z:", float(1 - res.square().sum() / fc.square().sum()))
show("sing. values of the linear map z->film", torch.linalg.svdvals(sol).numpy())

# 4. field-level: PCA of the field over codes at fixed points -> how many global field modes
x = (torch.rand(1, 1500, 3, generator=g) * 2 - 1)
x = x[:, x[0].norm(dim=-1) < 0.9]
with torch.no_grad():
    f = torch.cat([model(z[i:i + 64], x.expand(z[i:i + 64].shape[0], -1, -1)) for i in range(0, 512, 64)])
fcen = f - f.mean(0, keepdim=True)
s = torch.linalg.svdvals(fcen).numpy() / np.sqrt(f.shape[0] * f.shape[1])
show("PCA sing. values of field over 512 codes x points(|x|<0.9)", s)
# linear regression of field on z
sol2 = torch.linalg.lstsq(z[:512] - z[:512].mean(0), fcen).solution
res2 = fcen - (z[:512] - z[:512].mean(0)) @ sol2
print("share of field variance (across codes) explained by linear function of z:", float(1 - res2.square().sum() / fcen.square().sum()))
show("sing. values of linear map z->field", torch.linalg.svdvals(sol2).numpy())
# the active subspace: top right-singular vectors of the linear z->film map vs z->field map
U1, S1, V1 = torch.linalg.svd(sol.T, full_matrices=False)   # sol.T : 8192 x 32 ; V1 rows: directions in z
U2, S2, V2 = torch.linalg.svd(sol2.T, full_matrices=False)
k = 6
ov = torch.linalg.svdvals(V1[:k] @ V2[:k].T)
print(f"principal cosines between top-{k} z-subspaces of film map and field map:", np.array2string(ov.numpy(), precision=3))
# how much of W1 lives on that subspace
P = V2[:k]
print("share of mapping.0.weight energy in the top-6 field subspace:", float((W1 @ P.T).square().sum() / W1.square().sum()), "(chance 6/32=0.19)")
