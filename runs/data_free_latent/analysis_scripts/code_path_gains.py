"""Where on the way from the code to the shape do code directions get lost?

usage: code_path_gains.py <checkpoint> <tag>

The code enters the decoder through a small side network: code (unit, D) -> linear to 256 -> softplus ->
linear to 256 -> softplus -> linear to the scale and shift of every hidden unit of the main network.

1. The first of those linear layers: its singular values (a PCA of its weights), beside a freshly
   initialised layer of the same size.
2. The shared code directions, ranked by how much surface motion they cause (as in direction_loops.py), and
   for each one how strongly a step along it changes
     - the first layer's output,
     - the side network's last hidden layer,
     - the scales and shifts handed to the main network,
     - the surface itself.
   Each curve is shown relative to its own strongest direction.
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from arm_planner.data_free.decoder import ModulatedSphereDecoder
from arm_planner.data_free.field import field_and_grad
from arm_planner.data_free.latent import tangent_basis
from arm_planner.data_free.train import Trainer

ckpt, tag = sys.argv[1], sys.argv[2]
OUT = "/home/eric-luo/Downloads/pointcloud-motion-planning/figures"
trainer = Trainer.from_checkpoint(ckpt, device="cuda", tensorboard=False)
cfg, model = trainer.cfg, trainer.model.eval()
dim = int(cfg.latent_dim)
first = model.mapping[0].weight.detach().double()                       # 256 x D
torch.manual_seed(0)
fresh = ModulatedSphereDecoder(cfg).mapping[0].weight.detach().double()
sv = torch.linalg.svdvals(first).cpu().numpy()
sv0 = torch.linalg.svdvals(fresh).cpu().numpy()

# ------------------------------------------------------------------ shared directions by surface motion
gen = torch.Generator().manual_seed(7)
step, n_codes, chunk = 1e-3, 24, 8
codes = torch.nn.functional.normalize(torch.randn(n_codes, dim, generator=gen), dim=-1).cuda()
pooled = torch.zeros(dim, dim, device="cuda", dtype=torch.float64)
for k in range(n_codes):
    z = codes[k]
    x = (torch.rand(1, 5000, 3, generator=gen) * 2 - 1).cuda()
    mod = model.prepare(z[None])
    f, g = field_and_grad(model, mod, x, create_graph=False)
    xp = (x - f[..., None] * g).detach()
    fp, gp = field_and_grad(model, mod, xp, create_graph=False)
    keep = (fp[0].abs() < 3 * fp[0].abs().median().clamp_min(1e-4)) & (gp[0].norm(dim=-1) > 0.3)
    p, gn = xp[0][keep].detach(), gp[0][keep].detach()
    n_out = -gn / gn.norm(dim=-1, keepdim=True)
    T = tangent_basis(z)
    zs = torch.cat([z + step * T, z - step * T])
    zs = zs / zs.norm(dim=-1, keepdim=True)
    with torch.no_grad():
        vals = torch.cat([model(zs[i:i + chunk], p[None].expand(zs[i:i + chunk].shape[0], -1, -1))
                          for i in range(0, zs.shape[0], chunk)])
    half = dim - 1
    Dm = ((vals[:half] - vals[half:]) / (2 * step) / gn.norm(dim=-1)[None]).T
    eye = torch.eye(3, device="cuda")
    B = torch.cat([n_out @ eye,
                   torch.stack([(n_out * torch.linalg.cross(eye[i].expand_as(p), p)).sum(-1) for i in range(3)], 1),
                   (n_out * p).sum(-1, keepdim=True)], 1)
    R = (Dm - B @ torch.linalg.lstsq(B, Dm).solution).double()
    pooled += T.double().T @ (R.T @ R / p.shape[0]) @ T.double()
    del f, g, fp, gp
pooled /= n_codes
values, vectors = torch.linalg.eigh(pooled)
order = torch.argsort(values, descending=True)
motion = values[order].clamp_min(0).sqrt().cpu().numpy()                # surface motion per unit code change
directions = vectors[:, order].T                                        # D x D, double

# ------------------------------------------------------------------ how strongly each direction moves each stage
stages = {"first": torch.zeros(dim, dtype=torch.float64, device="cuda"),
          "hidden": torch.zeros(dim, dtype=torch.float64, device="cuda"),
          "film": torch.zeros(dim, dtype=torch.float64, device="cuda")}
eps = 1e-3
with torch.no_grad():
    for k in range(n_codes):
        z = codes[k].double()
        e = directions - (directions @ z)[:, None] * z                  # the part of each direction along the sphere
        keep_len = e.norm(dim=-1)                                       # shorter when the direction points along z
        plus = torch.nn.functional.normalize(z + eps * e, dim=-1).float()
        minus = torch.nn.functional.normalize(z - eps * e, dim=-1).float()

        def diff(fn):
            return ((fn(plus) - fn(minus)).double() / (2 * eps)).square().sum(-1)

        stages["first"] += diff(lambda c: model.mapping[0](c))
        stages["hidden"] += diff(lambda c: model.mapping(c))
        stages["film"] += diff(lambda c: model.film(model.mapping(c)))
gain = {name: (total / n_codes).sqrt().cpu().numpy() for name, total in stages.items()}

# the reverse view: the first layer's own principal directions, and the surface motion along each
_, s_first, vh = torch.linalg.svd(first, full_matrices=False)
along = torch.einsum("kd,de,ke->k", vh, pooled.cpu() if pooled.device != vh.device else pooled, vh).clamp_min(0).sqrt().cpu().numpy()
rank_a, rank_b = np.argsort(np.argsort(-s_first.cpu().numpy())), np.argsort(np.argsort(-along))
spearman = float(np.corrcoef(rank_a, rank_b)[0, 1])

share = lambda v, k: float((v[:k] ** 2).sum() / (v ** 2).sum())
print(f"{tag}: first layer {tuple(first.shape)}; singular values {sv[0]:.2f} (largest) to {sv[-1]:.2f} (smallest), "
      f"ratio {sv[0] / sv[-1]:.1f}; fresh layer {sv0[0]:.2f} to {sv0[-1]:.2f}, ratio {sv0[0] / sv0[-1]:.1f}")
print(f"  share of squared weight in the top 12 / 20 first-layer directions: {share(sv, 12):.2f} / {share(sv, 20):.2f} "
      f"(fresh: {share(sv0, 12):.2f} / {share(sv0, 20):.2f})")
print(f"  share of surface motion (squared) in the top 12 / 20 code directions: {share(motion, 12):.2f} / {share(motion, 20):.2f}")
for name, label in (("first", "first layer output"), ("hidden", "side network's last hidden layer"), ("film", "scales and shifts")):
    v = gain[name]
    print(f"  {label:34s} change per unit code step: strongest direction {v.max():.2f}, weakest {v.min():.2f}, "
          f"ratio {v.max() / v.min():.1f}; mean over surface-ranked directions 1-12 {v[:12].mean():.2f}, 13-32 {v[12:32].mean():.2f}, "
          f"33-64 {v[32:].mean():.2f}")
print(f"  surface motion: directions 1-12 {motion[:12].mean():.3f}, 13-32 {motion[12:32].mean():.3f}, 33-64 {motion[32:].mean():.3f}; "
      f"ratio strongest to weakest {motion[0] / motion[-1]:.0f}")
print(f"  first layer's own principal directions against surface motion along them: rank correlation {spearman:.2f}; "
      f"surface motion along its top 12: {along[:12].mean():.3f}, its last 32: {along[32:].mean():.3f}")

# ------------------------------------------------------------------ figure
INK, MUTED, GRIDC, PAPER = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
fig, (ax, bx) = plt.subplots(1, 2, figsize=(14.5, 5.6), gridspec_kw={"width_ratios": [1, 1.45]})
fig.patch.set_facecolor(PAPER)
idx = np.arange(1, dim + 1)
ax.plot(idx, sv, color=C1, lw=2, marker="o", ms=3.5, label="trained")
ax.plot(idx, sv0, color=MUTED, lw=1.6, ls="--", label="freshly initialised")
ax.set_ylim(0, max(sv.max(), sv0.max()) * 1.08)
ax.set_xlabel("principal direction of the first layer's weights, strongest first", fontsize=9.5, color=MUTED)
ax.set_ylabel("singular value (how much weight the direction gets)", fontsize=9.5, color=MUTED)
ax.set_title(f"First layer the code passes through (64 to 256): strongest direction\ngets {sv[0] / sv[-1]:.1f} times the weight of the weakest "
             f"(fresh layer: {sv0[0] / sv0[-1]:.1f} times)", fontsize=11, color=INK, loc="left")
ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower left")

rel = lambda v: v / v.max()
series = [("first layer's output", rel(gain["first"]), C1), ("side network's last hidden layer", rel(gain["hidden"]), C2),
          ("scales and shifts handed to the main network", rel(gain["film"]), C3), ("the surface itself", rel(motion), C4)]
for label, v, colour in series:
    bx.plot(idx, v, color=colour, lw=2, label=label)
    bx.text(dim + 0.8, v[-1], f"{v[-1]:.2f}", fontsize=8.5, color=INK, va="center")
bx.set_yscale("log"); bx.set_xlim(1, dim + 5)
bx.set_xlabel("code direction, ranked by how much it moves the surface (1 = most)", fontsize=9.5, color=MUTED)
bx.set_ylabel("change per unit code step, relative to the strongest direction", fontsize=9.5, color=MUTED)
bx.set_title("How strongly each code direction changes each stage\n(weakest direction relative to the strongest is printed at the right)",
             fontsize=11, color=INK, loc="left")
bx.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower left")
for cx in (ax, bx):
    cx.set_facecolor(PAPER); cx.grid(True, color=GRIDC, lw=0.6); cx.tick_params(colors=MUTED, labelsize=8.5, length=0)
    for side in cx.spines.values():
        side.set_visible(False)
fig.suptitle(f"{tag} (update {trainer.step_count}): where the code's directions are kept or lost", fontsize=13, color=INK, x=0.012, ha="left")
fig.tight_layout(rect=[0, 0, 1, 0.94])
path = f"{OUT}/data_free_code_path_gains_{tag}.png"
fig.savefig(path, dpi=115, facecolor=fig.get_facecolor())
print("saved", path)
