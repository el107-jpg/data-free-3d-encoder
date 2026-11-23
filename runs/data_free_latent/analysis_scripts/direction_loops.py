"""Walk once around the code sphere along every code direction and draw the shapes met on the way.

usage: direction_loops.py <checkpoint> <tag> [steps per loop] [resolution]

Directions: the surface motion each code direction causes (pose removed, as in evalrun.py) is measured at 24
random codes, the 64 x 64 matrices are averaged, and its eigenvectors are the directions, strongest first.
Loop for direction e: from one fixed random base code b, code(angle) = cos(angle) b + sin(angle) e', where e'
is e with its part along b removed. So every row starts at the same shape, is at the pure direction at 90
degrees, at the opposite of the base code at 180 and at minus the direction at 270.

Drawing: the field is evaluated on a grid turned to the viewing angle; a pixel is the first inside cell
along the line of sight, shaded by the field's slope there.
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

ckpt, tag = sys.argv[1], sys.argv[2]
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 16
res = int(sys.argv[4]) if len(sys.argv) > 4 else 80
OUT = "/home/eric-luo/Downloads/pointcloud-motion-planning/figures"
trainer = Trainer.from_checkpoint(ckpt, device="cuda", tensorboard=False)
cfg, model = trainer.cfg, trainer.model.eval()
dim = int(cfg.latent_dim)

# ------------------------------------------------------------------ directions
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
    Tm = T.double()
    pooled += Tm.T @ (R.T @ R / p.shape[0]) @ Tm
    del f, g, fp, gp
pooled /= n_codes
values, vectors = torch.linalg.eigh(pooled)
order = torch.argsort(values, descending=True)
rms = values[order].clamp_min(0).sqrt().cpu().numpy()
directions = vectors[:, order].T.float()                      # row i: direction i, strongest first
print("RMS surface motion per unit chord, strongest to weakest:", " ".join(f"{v:.3f}" for v in rms), flush=True)

# ------------------------------------------------------------------ view grid
elev, azim, extent = np.deg2rad(22.0), np.deg2rad(35.0), 0.9
w = np.array([np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)])     # towards the eye
u = np.cross([0.0, 0.0, 1.0], w); u /= np.linalg.norm(u)
v = np.cross(w, u)
axis = torch.linspace(-extent, extent, res, device="cuda")
a, b_, c = torch.meshgrid(axis, axis, axis, indexing="ij")
frame = torch.tensor(np.stack([u, v, w]), dtype=torch.float32, device="cuda")
pts = (a[..., None] * frame[0] + b_[..., None] * frame[1] + c[..., None] * frame[2]).reshape(1, -1, 3)
cell = 2 * extent / (res - 1)
light = torch.tensor([-0.45, 0.55, 0.70], device="cuda"); light = light / light.norm()
INK, MUTED, PAPER, BAND = "#1f1f1e", "#8a8985", np.array([252, 252, 251]) / 255, np.array([240, 239, 236]) / 255
BLUE = np.array([0x7f, 0xa8, 0xd6]) / 255


def render(code, paper):
    with torch.no_grad():
        F = torch.cat([model(code[None], pts[:, s:s + 200_000])[0] for s in range(0, pts.shape[1], 200_000)])
        F = F.reshape(res, res, res)
        gu, gv, gw = torch.gradient(F, spacing=cell)
        seen = F.flip(-1) > 0                                         # nearest the eye first
        first = seen.float().argmax(-1)
        idx = (res - 1 - first)[..., None]
        normal = -torch.stack([gu.gather(-1, idx), gv.gather(-1, idx), gw.gather(-1, idx)], -1)[..., 0, :]
        normal = normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        shade = 0.38 + 0.62 * (normal @ light).clamp_min(0.0)
        cover = (F.max(-1).values / cell + 0.5).clamp(0, 1)           # soft edge
        colour = shade[..., None] * torch.tensor(BLUE, dtype=torch.float32, device="cuda")
        back = torch.tensor(paper, dtype=torch.float32, device="cuda")
        img = cover[..., None] * colour + (1 - cover[..., None]) * back
    return img.permute(1, 0, 2).flip(0).cpu().numpy()                 # rows: up at the top


gen = torch.Generator().manual_seed(11)
base = torch.nn.functional.normalize(torch.randn(dim, generator=gen), dim=-1).cuda()
angles = np.arange(steps) * 2 * np.pi / steps
quarter = {steps // 4, 3 * steps // 4} if steps % 4 == 0 else set()
rows = []
for i in range(dim):
    e = directions[i] - (directions[i] @ base) * base
    e = e / e.norm()
    cells = [render(float(np.cos(t)) * base + float(np.sin(t)) * e, BAND if j in quarter else PAPER)
             for j, t in enumerate(angles)]
    rows.append(np.concatenate(cells, 1))
    if (i + 1) % 8 == 0:
        print(f"drawn {i + 1} of {dim} directions", flush=True)


def sheet(first, last, path):
    mosaic = np.concatenate(rows[first:last], 0)
    left, top, dpi = 190, 96, 100
    W, H = mosaic.shape[1] + left + 10, mosaic.shape[0] + top + 10
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    fig.patch.set_facecolor(PAPER)
    ax = fig.add_axes([left / W, 10 / H, mosaic.shape[1] / W, mosaic.shape[0] / H])
    ax.imshow(np.clip(mosaic, 0, 1), interpolation="nearest"); ax.set_axis_off()
    for r in range(first, last):
        y = 1 - (top + (r - first + 0.5) * res) / H
        fig.text(12 / W, y + 0.18 * res / H, f"direction {r + 1}", fontsize=10, color=INK, va="center")
        fig.text(12 / W, y - 0.18 * res / H, f"surface motion {rms[r]:.3f}", fontsize=8.5, color=MUTED, va="center")
    for j in range(steps):
        x = (left + (j + 0.5) * res) / W
        deg = 360.0 * j / steps
        note = {0: "base", steps // 4: "+direction", steps // 2: "opposite", 3 * steps // 4: "−direction"}.get(j, "") if steps % 4 == 0 else ""
        fig.text(x, 1 - (top - 26) / H, f"{deg:.0f}°", fontsize=8.5, color=MUTED, ha="center", va="center")
        if note:
            fig.text(x, 1 - (top - 10) / H, note, fontsize=8, color=INK, ha="center", va="center")
    fig.text(12 / W, 1 - 22 / H, f"{tag} (update {trainer.step_count}): one trip around the code sphere along each code direction, "
             f"directions {first + 1} to {last} of {dim}, strongest first", fontsize=12, color=INK, va="center")
    fig.text(12 / W, 1 - 44 / H, "Every row starts from the same random base code. Shaded columns are the pure direction and its negative; "
             "\"surface motion\" is RMS surface movement per unit of code change.", fontsize=9, color=MUTED, va="center")
    fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("saved", path, flush=True)


sheet(0, dim, f"{OUT}/data_free_direction_loops_{tag}_all.png")
per = 16
for page, first in enumerate(range(0, dim, per)):
    sheet(first, min(first + per, dim), f"{OUT}/data_free_direction_loops_{tag}_page{page + 1}.png")
