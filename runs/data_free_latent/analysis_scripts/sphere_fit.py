"""Can the plain main network (no base sphere added, no output bound) learn the field of a sphere?

usage: sphere_fit.py <config.json of the recipe> <out dir> <fourier bands> [updates=4000]

The decoder of the recipe (same widths, activation, Fourier bands, skip) with the code's scale and shift held
at zero, so the fitted network is the starting shape shared by all codes. Its raw output is the whole field.
Target: R - rho everywhere in the box (R = base_radius), value and slope. Points per update: half uniform in the
box, 40% within about 0.05 of the sphere, 10% within 0.2 of the centre (where the target has a cone tip).
Then, with the side network's usual starting spread switched on, how far apart random codes start and whether
their shapes are clean, next to the current decoder (sphere + bounded correction) at its start.
Writes <out dir>/sphere_fit.pt (decoder weights), sphere_fit.json and sphere_fit.png.
"""
import json
import math
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import ndimage

from arm_planner.data_free.config import DataFreeConfig
from arm_planner.data_free.decoder import ModulatedSphereDecoder, Modulation

cfg_path, out_dir, bands = sys.argv[1], sys.argv[2], int(sys.argv[3])
updates = int(sys.argv[4]) if len(sys.argv) > 4 else 4000
os.makedirs(out_dir, exist_ok=True)
raw = json.load(open(cfg_path))
cfg = DataFreeConfig.from_dict(raw)
cfg.fourier_bands = bands
torch.manual_seed(0)
dev = "cuda"
model = ModulatedSphereDecoder(cfg).to(dev)
R = float(cfg.base_radius)
L, W = len(model.hidden), model.width
zero = Modulation(torch.zeros(1, L, W, device=dev), torch.zeros(1, L, W, device=dev))
params = list(model.hidden.parameters()) + list(model.out.parameters())
opt = torch.optim.Adam(params, lr=3e-4)
warm = 100
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: (s + 1) / warm if s < warm else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(updates - warm, 1))))
gen = torch.Generator(device=dev).manual_seed(1)


def points(n):
    a, b = n // 2, (n * 4) // 10
    box = torch.rand(a, 3, device=dev, generator=gen) * 2 - 1
    u = torch.nn.functional.normalize(torch.randn(b, 3, device=dev, generator=gen), dim=-1)
    shell = u * (R + 0.05 * torch.randn(b, 1, device=dev, generator=gen))
    c = n - a - b
    v = torch.nn.functional.normalize(torch.randn(c, 3, device=dev, generator=gen), dim=-1)
    core = v * 0.2 * torch.rand(c, 1, device=dev, generator=gen) ** (1 / 3)
    return torch.cat([box, shell, core])


def field(x, mod=zero, graph=True):
    x = x.requires_grad_(True)
    f = model._net(mod, x[None].expand(mod.scale.shape[0], -1, -1))
    (g,) = torch.autograd.grad(f.sum(), x if mod.scale.shape[0] == 1 else x, create_graph=graph)
    return f, g


history = []
t0 = time.time()
for step in range(updates):
    x = points(16384)
    rho = x.norm(dim=-1).clamp_min(1e-6)
    f, g = field(x)
    target, tgrad = R - rho, -x / rho[:, None]
    value = (f[0] - target).abs().mean()
    slope = (g - tgrad).square().sum(-1).mean()
    loss = value + 0.1 * slope
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    sched.step()
    if step % 250 == 0 or step == updates - 1:
        history.append((step, float(value), float(slope)))
        print(f"update {step:5d}  value error {float(value):.5f}  slope mismatch {float(slope):.5f}  {time.time() - t0:.0f} s", flush=True)

# ------------------------------------------------------------------ evaluation of the fit
res = {"fourier_bands": bands, "updates": updates, "seconds": time.time() - t0}
ev = torch.Generator(device=dev).manual_seed(2)
x = torch.rand(400000, 3, device=dev, generator=ev) * 2 - 1
x = torch.cat([x, torch.nn.functional.normalize(torch.randn(100000, 3, device=dev, generator=ev), dim=-1)
               * (R + 0.03 * torch.randn(100000, 1, device=dev, generator=ev)),
               torch.nn.functional.normalize(torch.randn(50000, 3, device=dev, generator=ev), dim=-1)
               * 0.1 * torch.rand(50000, 1, device=dev, generator=ev)])
fs, gs = [], []
for s in range(0, x.shape[0], 50000):
    f, g = field(x[s:s + 50000].clone(), graph=False)
    fs.append(f[0].detach()); gs.append(g.detach())
f, g = torch.cat(fs), torch.cat(gs)
rho = x.norm(dim=-1)
err, serr = (f - (R - rho)).abs(), (g.norm(dim=-1) - 1).abs()
regions = {"within 0.03 of the surface": (rho - R).abs() < 0.03, "centre (radius < 0.1)": rho < 0.1,
           "rest of the ball (radius < 0.9)": (rho < 0.9) & ((rho - R).abs() >= 0.03) & (rho >= 0.1),
           "outside radius 0.9 (corners of the box)": rho >= 0.9}
res["regions"] = {}
for name, m in regions.items():
    res["regions"][name] = {"value_error_mean": float(err[m].mean()), "value_error_max": float(err[m].max()),
                            "slope_error_mean": float(serr[m].mean()), "slope_error_p99": float(serr[m].quantile(0.99))}
# where is the surface? slide points onto f = 0 and read their radius
u = torch.nn.functional.normalize(torch.randn(20000, 3, device=dev, generator=ev), dim=-1) * R
for _ in range(6):
    fv, gv = field(u.clone(), graph=False)
    u = (u - (fv[0] / gv.square().sum(-1).clamp_min(1e-6))[:, None] * gv).detach()
radius = u.norm(dim=-1)
res["surface_radius"] = {"mean": float(radius.mean()), "std": float(radius.std()),
                         "max_deviation": float((radius - R).abs().max())}
# stray pieces on a 128 grid
n = 128
axis = torch.linspace(-1, 1, n, device=dev)
grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)


def grid_field(mod):
    out = []
    with torch.no_grad():
        for i in range(mod.scale.shape[0]):
            one = Modulation(mod.scale[i:i + 1], mod.shift[i:i + 1])
            out.append(torch.cat([model._net(one, grid[None, s:s + 131072]) for s in range(0, grid.shape[0], 131072)],
                                 dim=1).reshape(n, n, n).cpu().numpy())
    return np.stack(out)


def pieces(vol):
    inside, _ = ndimage.label(vol > 0)
    outside, k_out = ndimage.label(vol <= 0)
    return int(inside.max()), int(outside.max())


fgrid = grid_field(zero)[0]
res["grid_pieces_inside_outside"] = pieces(fgrid)
res["history"] = history
print(json.dumps({k: v for k, v in res.items() if k != "history"}, indent=1), flush=True)

# ------------------------------------------------------------------ codes switched on
codes = torch.nn.functional.normalize(torch.randn(32, int(cfg.latent_dim), generator=torch.Generator().manual_seed(3)), dim=-1).to(dev)
with torch.no_grad():
    mod = model.prepare(codes)
    current = ModulatedSphereDecoder(DataFreeConfig.from_dict(raw)).to(dev)   # today's start, for comparison
    probe = (torch.rand(1, 20000, 3, device=dev, generator=ev) * 2 - 1) * 0.9
    cur = current(codes, probe.expand(32, -1, -1))
    cur_spread = float((cur[:, None] - cur[None]).abs().mean())
res["current_start_mean_difference_between_codes"] = cur_spread
res["codes"] = {}
for factor in (1.0, 0.3, 0.1, 0.03):
    m = Modulation(mod.scale * factor, mod.shift * factor)
    with torch.no_grad():
        fc = model._net(m, probe.expand(32, -1, -1))
    spread = float((fc[:, None] - fc[None]).abs().mean())
    off = float((fc - (R - probe.norm(dim=-1))).abs().mean())
    vols = grid_field(m)
    clean = sum(1 for v in vols if pieces(v) == (1, 1))
    empty = sum(1 for v in vols if (v > 0).sum() == 0)
    res["codes"][f"starting spread x{factor}"] = {"film_std": 0.1 * factor * float(cfg.film_init_std) / 0.1,
                                                  "mean_difference_between_codes": spread,
                                                  "mean_difference_from_sphere": off, "clean_of_32": clean, "empty": empty}
    print(f"code scale and shift at {factor} x the usual starting spread: mean difference between codes {spread:.4f} "
          f"(today's start: {cur_spread:.4f}); from the sphere {off:.4f}; clean {clean}/32, empty {empty}", flush=True)

torch.save(model.state_dict(), os.path.join(out_dir, "sphere_fit.pt"))
json.dump(res, open(os.path.join(out_dir, "sphere_fit.json"), "w"), indent=1)

# ------------------------------------------------------------------ picture: slice z = 0
k = n // 2
xs = axis.cpu().numpy()
slab = fgrid[:, :, k]
X, Y = np.meshgrid(xs, xs, indexing="ij")
truth = R - np.sqrt(X ** 2 + Y ** 2)
with torch.enable_grad():
    pts = torch.stack([torch.tensor(X), torch.tensor(Y), torch.zeros_like(torch.tensor(X))], -1).reshape(-1, 3).float().to(dev)
    _, gsl = field(pts.clone(), graph=False)
slope_slab = (gsl.norm(dim=-1) - 1).reshape(n, n).cpu().numpy()
INK, MUTED, PAPER = "#0b0b0b", "#52514e", "#fcfcfb"
fig, axs = plt.subplots(1, 3, figsize=(15, 4.8))
fig.patch.set_facecolor(PAPER)
lim = max(np.abs(slab - truth).max(), 1e-4)
im = axs[0].imshow((slab - truth).T, origin="lower", extent=(-1, 1, -1, 1), cmap="RdBu_r", vmin=-lim, vmax=lim)
axs[0].contour(X, Y, slab, levels=[0], colors=INK, linewidths=1)
axs[0].set_title(f"field minus the sphere's true field (z = 0)\nblack: the fitted surface", fontsize=10, color=INK, loc="left")
fig.colorbar(im, ax=axs[0], fraction=0.046)
lim2 = max(np.abs(slope_slab).max(), 1e-3)
im2 = axs[1].imshow(slope_slab.T, origin="lower", extent=(-1, 1, -1, 1), cmap="RdBu_r", vmin=-lim2, vmax=lim2)
axs[1].set_title("slope minus 1 (z = 0)", fontsize=10, color=INK, loc="left")
fig.colorbar(im2, ax=axs[1], fraction=0.046)
h = np.array(history)
axs[2].semilogy(h[:, 0], h[:, 1], color="#2a78d6", lw=2, label="value error (mean |f - true|)")
axs[2].semilogy(h[:, 0], h[:, 2], color="#eb6834", lw=2, label="slope mismatch (mean squared)")
axs[2].set_xlabel("update", fontsize=9, color=MUTED); axs[2].legend(frameon=False, fontsize=9)
axs[2].set_title("fit during training", fontsize=10, color=INK, loc="left")
for a in axs:
    a.set_facecolor(PAPER); a.tick_params(colors=MUTED, labelsize=8)
fig.suptitle(f"Plain main network fitted to a sphere of radius {R} ({bands} Fourier bands, {updates} updates)", fontsize=12, color=INK, x=0.01, ha="left")
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "sphere_fit.png"), dpi=110, facecolor=PAPER)
print("saved", out_dir)
