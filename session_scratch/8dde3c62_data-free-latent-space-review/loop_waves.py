"""Does the shape go round once per trip around the code sphere, or does it wave (repeat) on the way?

usage: loop_waves.py <out.png> <checkpoint> <label> [<checkpoint> <label> ...]

Loops: from one fixed random base code b, code(angle) = cos(angle) b + sin(angle) e for several shared code
directions e (ranked by surface motion, as in direction_loops.py), plus a few loops through two random codes.
Each loop is sampled at 128 angles and the field is evaluated at 4,000 fixed points inside the bound radius.

 1. Return curve: mean |field difference| between the shapes at two angles, as a function of how far apart
    the angles are. One trip with no waves rises to a single peak near 180 degrees and comes back. A dip back
    towards zero before 360 would be a repeat (a wave of that period); a value near zero AT 180 would mean a
    code and its opposite give the same shape.
 2. Harmonics: the field at each point as a function of the angle, split into waves of 1, 2, 3, ... cycles per
    trip; the share of the change carried by each.
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from arm_planner.data_free.field import field_and_grad
from arm_planner.data_free.latent import tangent_basis
from arm_planner.data_free.train import Trainer

out_path = sys.argv[1]
pairs = list(zip(sys.argv[2::2], sys.argv[3::2]))
N, RANKS, RANDOM = 128, (1, 4, 8, 16, 32, 48, 64), 4
INK, MUTED, GRIDC, PAPER = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
RAMP = ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#b7d3f6"]      # strongest direction darkest
C1, C2 = "#2a78d6", "#eb6834"
angles = torch.arange(N) * (2 * np.pi / N)


def shared_directions(model, dim):
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
    values, vectors = torch.linalg.eigh(pooled / n_codes)
    order = torch.argsort(values, descending=True)
    return vectors[:, order].T.float()


def measure(ckpt):
    trainer = Trainer.from_checkpoint(ckpt, device="cuda", tensorboard=False)
    cfg, model = trainer.cfg, trainer.model.eval()
    dim, bound = int(cfg.latent_dim), float(cfg.bound_radius)
    directions = shared_directions(model, dim)
    gen = torch.Generator().manual_seed(11)
    base = torch.nn.functional.normalize(torch.randn(dim, generator=gen), dim=-1).cuda()
    x = torch.randn(40000, 3, generator=gen)
    x = (x / x.norm(dim=-1, keepdim=True) * bound * torch.rand(40000, 1, generator=gen) ** (1 / 3))[:4000].cuda()
    loops = []
    for rank in RANKS:
        e = directions[rank - 1] - (directions[rank - 1] @ base) * base
        loops.append((f"direction {rank}", base, e / e.norm()))
    for k in range(RANDOM):
        a = torch.nn.functional.normalize(torch.randn(dim, generator=gen), dim=-1).cuda()
        b = torch.nn.functional.normalize(torch.randn(dim, generator=gen), dim=-1).cuda()
        b = b - (b @ a) * a
        loops.append((f"random loop {k + 1}", a, b / b.norm()))
    result = []
    ang = angles.cuda()
    for name, a, b in loops:
        codes = torch.cos(ang)[:, None] * a + torch.sin(ang)[:, None] * b
        with torch.no_grad():
            F = torch.cat([model(codes[s:s + 16], x[None].expand(codes[s:s + 16].shape[0], -1, -1)) for s in range(0, N, 16)])
            inside = (F > 0).float()
            # mean |field difference| and solid overlap between angle t and angle t + shift, averaged over t
            gap = torch.stack([(F - F.roll(-s, 0)).abs().mean() for s in range(N)]).cpu().numpy()
            both = torch.stack([(inside * inside.roll(-s, 0)).sum(1) / ((inside + inside.roll(-s, 0)) > 0).float().sum(1).clamp_min(1)
                                for s in range(N)]).mean(1).cpu().numpy()
            power = torch.fft.rfft(F.double(), dim=0).abs().square().sum(1)[1:].cpu().numpy()        # harmonics 1 .. N/2
        peak = int(gap.argmax())
        half = gap[: N // 2 + 1]
        dips = sum(1 for j in range(2, N // 2 - 1) if half[j] < half[j - 1] and half[j] < half[j + 1]
                   and half[:j].max() - half[j] > 0.05 * gap.max())
        result.append({"name": name, "gap": gap, "overlap": both, "share": power / power.sum(), "peak_deg": 360.0 * peak / N,
                       "dips": dips, "at180": gap[N // 2] / gap.max(), "overlap180": both[N // 2]})
    return trainer.step_count, result


data = []
for ckpt, label in pairs:
    step, res = measure(ckpt)
    data.append((label, step, res))
    print(f"\n{label} (update {step})")
    for r in res:
        s = r["share"]
        print(f"  {r['name']:15s} largest difference at {r['peak_deg']:5.1f} deg apart; difference at 180 deg = {r['at180']:.2f} of the largest "
              f"(solid overlap there {r['overlap180']:.2f}); repeats on the way: {r['dips']}; "
              f"share of change in 1 / 2 / 3 / 4+ cycles per trip: {s[0]:.2f} / {s[1]:.2f} / {s[2]:.2f} / {s[3:].sum():.3f}")
    mean = np.mean([r["share"] for r in res], 0)
    print(f"  all loops: 1 cycle {mean[0]:.2f}, 2 cycles {mean[1]:.2f}, 3 cycles {mean[2]:.2f}, 4 to 8 cycles {mean[3:8].sum():.3f}, more than 8 {mean[8:].sum():.4f}")

cols = len(data) + 1
fig, axes = plt.subplots(1, cols, figsize=(5.6 * cols, 5.4))
fig.patch.set_facecolor(PAPER)
deg = np.arange(N + 1) * 360.0 / N
for ax, (label, step, res) in zip(axes, data):
    for r in res[len(RANKS):]:
        ax.plot(deg, np.append(r["gap"], r["gap"][0]) / r["gap"].max(), color=MUTED, lw=1.0, ls="--")
    for colour, r in zip(RAMP, res[:len(RANKS)]):
        ax.plot(deg, np.append(r["gap"], r["gap"][0]) / r["gap"].max(), color=colour, lw=1.8, label=r["name"])
    ax.plot([], [], color=MUTED, lw=1.0, ls="--", label="loops through two random codes")
    ax.set_xlim(0, 360); ax.set_ylim(0, 1.05); ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_xlabel("angle between two codes on the same loop (degrees)", fontsize=9.5, color=MUTED)
    ax.set_ylabel("shape difference, relative to the loop's largest", fontsize=9.5, color=MUTED)
    ax.set_title(f"{label} (update {step})\nhow different two shapes on a loop are", fontsize=11, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower center", ncol=2)
bx = axes[-1]
ks = np.arange(1, 9)
width = 0.8 / len(data)
for j, ((label, step, res), colour) in enumerate(zip(data, (C1, C2, "#1baf7a", "#eda100"))):
    mean = np.mean([r["share"] for r in res], 0)
    bx.bar(ks + (j - (len(data) - 1) / 2) * width, mean[:8], width=width * 0.92, color=colour, label=label)
    bx.text(1 + (j - (len(data) - 1) / 2) * width, mean[0] + 0.015, f"{mean[0]:.2f}", ha="center", fontsize=8.5, color=INK)
bx.set_xticks(ks); bx.set_ylim(0, 1.0)
bx.set_xlabel("cycles per trip around the loop", fontsize=9.5, color=MUTED)
bx.set_ylabel("share of the field's change along the loop", fontsize=9.5, color=MUTED)
bx.set_title("Waves along the loops, averaged over all loops\n(1 = once round; higher = repeating on the way)", fontsize=11, color=INK, loc="left")
bx.legend(frameon=False, fontsize=9, labelcolor=INK)
for cx in axes:
    cx.set_facecolor(PAPER); cx.grid(True, color=GRIDC, lw=0.6); cx.set_axisbelow(True); cx.tick_params(colors=MUTED, labelsize=8.5, length=0)
    for side in cx.spines.values():
        side.set_visible(False)
fig.tight_layout()
fig.savefig(out_path, dpi=115, facecolor=fig.get_facecolor())
print("\nsaved", out_path)
