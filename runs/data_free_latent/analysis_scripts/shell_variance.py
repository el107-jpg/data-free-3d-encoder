"""Shapes of probe codes, and how much the surface moves along each direction of the latent shell.

Local: at each probe code, the 31 tangent directions are rotated (SVD) into the directions of
most to least surface motion; the squared singular values are the variance of surface motion
along those directions (RMS outward displacement per unit chord, squared).
Global: the same motion expressed in the 32 ambient code coordinates and averaged over codes;
its eigenvalues show whether some fixed directions of code space matter more than others.
Both are given raw and after removing translation, rotation and resize.
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from arm_planner.data_free.field import field_and_grad
from arm_planner.data_free.latent import tangent_basis
from arm_planner.data_free.train import Trainer

ckpt, tag, label = sys.argv[1], sys.argv[2], sys.argv[3]
OUT = "/home/eric-luo/Downloads"
trainer = Trainer.from_checkpoint(ckpt, device="cuda", tensorboard=False)
model = trainer.model.eval()
codes = trainer.bank.codes
step_count = trainer.step_count

# ------------------------------------------------------------------ slices
count, res = 8, 224
axis = torch.linspace(-1, 1, res, device="cuda")
ga, gb = torch.meshgrid(axis, axis, indexing="xy")
zero = torch.zeros_like(ga)
planes = [("x = 0", torch.stack([zero, ga, gb], -1)), ("y = 0", torch.stack([ga, zero, gb], -1)),
          ("z = 0", torch.stack([ga, gb, zero], -1))]
fig, axes = plt.subplots(3, count, figsize=(2.3 * count, 7.4))
theta = np.linspace(0, 2 * np.pi, 200)
a = axis.cpu().numpy()
for r, (name, pts) in enumerate(planes):
    with torch.no_grad():
        f = torch.cat([model(codes[i:i + 4], pts.reshape(1, -1, 3).expand(4, -1, -1)) for i in range(0, count, 4)])
    f = f.reshape(count, res, res).cpu().numpy()
    for i in range(count):
        ax = axes[r, i]
        ax.contourf(a, a, f[i], levels=[0, 10], colors=["#9ecae1"])
        ax.contour(a, a, f[i], levels=[0], colors="#08519c", linewidths=1.6)
        ax.plot(0.5 * np.cos(theta), 0.5 * np.sin(theta), color="#d95f02", lw=0.7, ls="--")
        ax.set_aspect("equal"); ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title(f"probe code {i}", fontsize=9)
        if i == 0:
            ax.set_ylabel(f"slice {name}", fontsize=9)
fig.suptitle(f"{label}, step {step_count}: three slices of 8 probe codes\n"
             "Blue: shape and its surface; dashed orange: initial sphere R = 0.5", fontsize=11)
fig.tight_layout()
fig.savefig(f"{OUT}/data_free_shapes_{tag}.png", dpi=100)
plt.close(fig)

# ------------------------------------------------------- variance along latent directions
gen = torch.Generator().manual_seed(7)
step, n_codes = 1e-3, 24
local_raw, local_inv = [], []
gram_raw = torch.zeros(32, 32, device="cuda", dtype=torch.float64)
gram_inv = torch.zeros_like(gram_raw)
shares = []
for k in range(n_codes):
    z = codes[k]
    x = (torch.rand(1, 5000, 3, generator=gen) * 2 - 1).cuda()
    mod = model.prepare(z[None])
    f, g = field_and_grad(model, mod, x, create_graph=False)
    xp = (x - f[..., None] * g).detach()
    fp, gp = field_and_grad(model, mod, xp, create_graph=False)
    keep = (fp[0].abs() < 3 * fp[0].abs().median().clamp_min(1e-4)) & (gp[0].norm(dim=-1) > 0.3)
    p, gn = xp[0][keep], gp[0][keep]
    n_out = -gn / gn.norm(dim=-1, keepdim=True)
    T = tangent_basis(z)                                              # 31 x 32
    zs = torch.cat([z + step * T, z - step * T])
    zs = zs / zs.norm(dim=-1, keepdim=True)
    with torch.no_grad():
        vals = torch.cat([model(zs[i:i + 8], p[None].expand(zs[i:i + 8].shape[0], -1, -1))
                          for i in range(0, zs.shape[0], 8)])
    D = ((vals[:31] - vals[31:]) / (2 * step) / gn.norm(dim=-1)[None]).T   # P x 31
    eye = torch.eye(3, device="cuda")
    B = torch.cat([n_out @ eye,
                   torch.stack([(n_out * torch.linalg.cross(eye[i].expand_as(p), p)).sum(-1) for i in range(3)], 1),
                   (n_out * p).sum(-1, keepdim=True)], 1)
    R = D - B @ torch.linalg.lstsq(B, D).solution                    # pose removed
    scale = 1.0 / np.sqrt(p.shape[0])
    for M, store, gram in ((D, local_raw, gram_raw), (R, local_inv, gram_inv)):
        s = torch.linalg.svdvals(M * scale)
        store.append((s ** 2).cpu().numpy())
        J = (M * scale).double() @ T.double()                         # P x 32, in ambient code coordinates
        gram += J.T @ J / n_codes
    shares.append(float(R.square().sum() / D.square().sum()))
local_raw, local_inv = np.array(local_raw), np.array(local_inv)
glob_raw = np.sort(np.linalg.eigvalsh(gram_raw.cpu().numpy()))[::-1][:31]
glob_inv = np.sort(np.linalg.eigvalsh(gram_inv.cpu().numpy()))[::-1][:31]


def summary(v):
    v = np.maximum(v, 0)
    p = v / v.sum()
    eff = float(np.exp(-(p * np.log(p + 1e-30)).sum()))                  # effective number of directions
    c = np.cumsum(p)
    return {"total": float(v.sum()), "top1_share": float(p[0]), "top3_share": float(c[2]),
            "n_for_90pct": int(np.searchsorted(c, 0.9) + 1), "effective_directions": eff,
            "rms_top": float(np.sqrt(v[0])), "rms_median": float(np.sqrt(np.median(v))),
            "ratio_top_to_median": float(v[0] / max(np.median(v), 1e-30))}


res_json = {"label": label, "step": step_count, "codes": n_codes,
            "deformation_share_of_motion": float(np.mean(shares)),
            "local_raw": summary(local_raw.mean(0)), "local_pose_removed": summary(local_inv.mean(0)),
            "global_raw": summary(glob_raw), "global_pose_removed": summary(glob_inv),
            "local_pose_removed_variances": local_inv.mean(0).tolist(),
            "global_pose_removed_variances": glob_inv.tolist()}
print(json.dumps({k: v for k, v in res_json.items() if not k.endswith("variances")}, indent=1))
json.dump(res_json, open(f"/home/eric-luo/.claude/jobs/8dde3c62/tmp/shell_variance_{tag}.json", "w"), indent=1)

# chart: variance per direction, sorted; grey = all motion, blue = pose removed
idx = np.arange(1, 32)
fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.3), sharey=True)
for ax, raw, inv, title in ((axs[0], local_raw.mean(0), local_inv.mean(0),
                             f"Local: directions sorted at each code (mean of {n_codes} codes)"),
                            (axs[1], glob_raw, glob_inv,
                             "Global: fixed directions of code space (averaged over codes)")):
    ax.bar(idx - 0.2, raw, width=0.38, color="#b8b8b3", label="all surface motion")
    ax.bar(idx + 0.2, inv, width=0.38, color="#1f4f9c", label="after removing move / turn / resize")
    ax.set_yscale("log"); ax.set_xlabel("latent direction, sorted from most to least motion")
    ax.set_title(title, fontsize=10); ax.grid(axis="y", color="#e3e3df", lw=0.6); ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
axs[0].set_ylabel("variance of surface motion\n(squared RMS displacement per unit chord)")
axs[0].legend(frameon=False, fontsize=9)
fig.suptitle(f"{label}, step {step_count}: how much the surface moves along each direction of the latent shell",
             fontsize=11)
fig.tight_layout()
fig.savefig(f"{OUT}/data_free_shell_variance_{tag}.png", dpi=110)
print("saved", f"{OUT}/data_free_shapes_{tag}.png", f"{OUT}/data_free_shell_variance_{tag}.png")
