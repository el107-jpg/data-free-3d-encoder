"""Do many codes give the same shape? Gradient descent from many random starts toward one target.

usage: redundancy_paths.py <checkpoint> <tag> [starts] [steps]

Targets: three of the model's OWN shapes (decoded from random codes, so an exact code is known to exist) and
three Thingi objects the model never trained on. For each target, `starts` random unit codes are optimised
with the decoder frozen (same loss as the held-out fit test: mean |field difference| inside the bound radius).

Then, for the best start b and every other start j:
 - overlap of the two final solids (same shape or not), and the cosine between the two final codes;
 - the solids met on the straight path on the code sphere from b to j, compared with b's solid. A path that
   keeps the shape all the way is a flat valley (the codes differ only in directions that do nothing);
   a path that leaves the shape and comes back is two separate codes for one shape.

Writes figures/data_free_redundancy_paths_<tag>.png and prints a summary per target.
"""
import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from arm_planner.data_free import losses as L
from arm_planner.data_free.train import Trainer

ckpt, tag = sys.argv[1], sys.argv[2]
S = int(sys.argv[3]) if len(sys.argv) > 3 else 16
steps = int(sys.argv[4]) if len(sys.argv) > 4 else 500
OUT = "/home/eric-luo/Downloads/pointcloud-motion-planning/figures"
DS = "/home/eric-luo/Downloads/pointcloud-motion-planning/datasets/"
dev = "cuda"
trainer = Trainer.from_checkpoint(ckpt, device=dev, tensorboard=False)
cfg, model = trainer.cfg, trainer.model.eval()
for p in model.parameters():
    p.requires_grad_(False)
dim, bound = int(cfg.latent_dim), float(cfg.bound_radius)
unit = lambda c: torch.nn.functional.normalize(c, dim=-1)
gen = torch.Generator().manual_seed(31)
test_x = (torch.rand(20000, 3, generator=gen) * 2 - 1).to(dev)

# ------------------------------------------------------------------ targets
targets = []
own = unit(torch.randn(3, dim, generator=gen)).to(dev)
for k in range(3):
    code = own[k]
    targets.append({"name": f"own shape {k + 1}", "own": code,
                    "value": (lambda x, code=code: model(code[None], x)[0])})
pool = torch.load(DS + "data_free_anchors_20261112_v3/eligible.pt", map_location="cpu", weights_only=False)
skip = set(int(i) for i in torch.load(DS + "data_free_anchors_20261111_v1/anchors.pt", map_location="cpu",
                                      weights_only=False)["file_ids"])
skip |= set(int(i) for i in torch.load(DS + "data_free_anchors_20261112_v3/anchors.pt", map_location="cpu",
                                       weights_only=False)["file_ids"])
free = [k for k, i in enumerate(pool["file_ids"]) if int(i) not in skip]
pick_gen = torch.Generator().manual_seed(0)
for k in [free[int(j)] for j in torch.randperm(len(free), generator=pick_gen)[:3]]:
    grid = pool["grid"][k:k + 1].float().to(dev)
    name = str(pool["names"][k]).replace("&#039;", "'")[:22] if "names" in pool else f"object {k}"
    targets.append({"name": f"unseen: {name}", "own": None,
                    "value": (lambda x, grid=grid: L.sample_grid(grid, x)[0])})

# ------------------------------------------------------------------ drawing (field on a grid turned to the view)
res, extent = 72, 0.9
elev, azim = np.deg2rad(22.0), np.deg2rad(35.0)
w = np.array([np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)])
u = np.cross([0.0, 0.0, 1.0], w); u /= np.linalg.norm(u)
v = np.cross(w, u)
axis = torch.linspace(-extent, extent, res, device=dev)
a, b_, c_ = torch.meshgrid(axis, axis, axis, indexing="ij")
frame = torch.tensor(np.stack([u, v, w]), dtype=torch.float32, device=dev)
view_pts = (a[..., None] * frame[0] + b_[..., None] * frame[1] + c_[..., None] * frame[2]).reshape(1, -1, 3)
cell = 2 * extent / (res - 1)
light = torch.tensor([-0.45, 0.55, 0.70], device=dev); light = light / light.norm()
PAPER = np.array([252, 252, 251]) / 255
INK, MUTED, GRID = "#1f1f1e", "#8a8985", "#e4e3df"
BLUE, TAN = np.array([0x7f, 0xa8, 0xd6]) / 255, np.array([0xc9, 0xa2, 0x6b]) / 255
LINE = "#3f7fc4"


def render(value_fn, colour):
    with torch.no_grad():
        F = torch.cat([value_fn(view_pts[:, s:s + 200_000]) for s in range(0, view_pts.shape[1], 200_000)])
        F = F.reshape(res, res, res)
        gu, gv, gw = torch.gradient(F, spacing=cell)
        first = (F.flip(-1) > 0).float().argmax(-1)
        idx = (res - 1 - first)[..., None]
        normal = -torch.stack([gu.gather(-1, idx), gv.gather(-1, idx), gw.gather(-1, idx)], -1)[..., 0, :]
        normal = normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        shade = 0.38 + 0.62 * (normal @ light).clamp_min(0.0)
        cover = (F.max(-1).values / cell + 0.5).clamp(0, 1)
        img = cover[..., None] * shade[..., None] * torch.tensor(colour, dtype=torch.float32, device=dev) \
            + (1 - cover[..., None]) * torch.tensor(PAPER, dtype=torch.float32, device=dev)
    return np.clip(img.permute(1, 0, 2).flip(0).cpu().numpy(), 0, 1)


def solids(codes):
    with torch.no_grad():
        return torch.cat([model(codes[s:s + 16], test_x.expand(codes[s:s + 16].shape[0], -1, -1)) > 0
                          for s in range(0, codes.shape[0], 16)])


def overlap(one, many):
    """IoU of one solid (M,) with each row of many (K, M)."""
    inter = (many & one).sum(1).float()
    return inter / (many | one).sum(1).clamp_min(1).float()


# ------------------------------------------------------------------ fits
fracs = torch.linspace(0, 1, 11, device=dev)
results = []
for t in targets:
    with torch.no_grad():
        want = t["value"](test_x[None]) > 0
    codes = unit(torch.randn(S, dim, generator=gen)).to(dev).requires_grad_(True)
    opt = torch.optim.Adam([codes], lr=0.03)
    history = []
    for step in range(steps + 1):
        if step % 10 == 0:
            history.append(overlap(want, solids(unit(codes.detach()))).cpu().numpy())
        if step == steps:
            break
        for group in opt.param_groups:
            group["lr"] = 0.002 + 0.028 * 0.5 * (1 + math.cos(math.pi * step / steps))
        x = (torch.rand(1, 2048, 3, generator=gen) * 2 - 1).to(dev)
        inside = (x.square().sum(-1) < bound ** 2).float()
        with torch.no_grad():
            goal = t["value"](x)
        f = model(unit(codes), x.expand(S, -1, -1))
        loss = (((f - goal[None]).abs() * inside).sum(1) / inside.sum()).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    final = unit(codes.detach())
    shape = solids(final)
    fit = overlap(want, shape)
    order = torch.argsort(fit, descending=True)
    best = int(order[0])
    cosine = final @ final[best]
    same = overlap(shape[best], shape)
    paths = []
    angle = torch.arccos(cosine.clamp(-1, 1))
    for j in range(S):
        if float(angle[j]) < 1e-4:
            way = final[best][None].expand(len(fracs), -1)
        else:
            way = (torch.sin((1 - fracs)[:, None] * angle[j]) * final[best] + torch.sin(fracs[:, None] * angle[j]) * final[j]) \
                / torch.sin(angle[j])
        met = solids(unit(way))
        paths.append(torch.maximum(overlap(shape[best], met), overlap(shape[j], met)).cpu().numpy() if False
                     else np.stack([overlap(shape[best], met).cpu().numpy(), overlap(shape[j], met).cpu().numpy()]))
    paths = np.stack(paths)                                   # S x 2 x 11
    keep = np.maximum(paths[:, 0], paths[:, 1]).min(1)        # lowest "still one of the two end shapes" on the path
    others = [j for j in range(S) if j != best]
    same_shape = [j for j in others if float(same[j]) >= 0.90]
    flat = [j for j in same_shape if keep[j] >= 0.85]
    apart = [j for j in same_shape if keep[j] < 0.85]
    line = (f"{t['name']:32s} best overlap with target {float(fit[best]):.2f}; starts within 0.03 of it: "
            f"{int((fit >= fit[best] - 0.03).sum())} of {S}; other starts ending at the same shape as the best "
            f"(overlap >= 0.90): {len(same_shape)} of {S - 1}")
    if same_shape:
        cs = torch.stack([cosine[j] for j in same_shape])
        line += (f", their code cosine to the best {float(cs.min()):.2f} to {float(cs.max()):.2f} (median {float(cs.median()):.2f}); "
                 f"joined to it by a shape-keeping path: {len(flat)}, separated: {len(apart)}")
    if t["own"] is not None:
        line += f"; cosine of the best code to the code that made the target {float(final[best] @ t['own']):.2f}"
    print(line, flush=True)
    show = [int(order[r]) for r in (0, 1, S // 3, 2 * S // 3, S - 1)]
    results.append({"t": t, "history": np.stack(history), "fit": fit.cpu().numpy(), "cosine": cosine.cpu().numpy(),
                    "same": same.cpu().numpy(), "paths": paths, "best": best, "show": show,
                    "target_img": render(t["value"], TAN),
                    "imgs": [render((lambda x, code=final[j]: model(code[None], x)[0]), BLUE) for j in show]})

# ------------------------------------------------------------------ figure
rows, cols = len(results), 8
fig = plt.figure(figsize=(2.25 * cols + 1.2, 2.45 * rows + 1.3))
fig.patch.set_facecolor(PAPER)
grid = fig.add_gridspec(rows, cols, left=0.035, right=0.985, top=0.905, bottom=0.05, wspace=0.28, hspace=0.42,
                        width_ratios=[1, 1, 1, 1, 1, 1, 1.35, 1.35])
for r, R in enumerate(results):
    ax = fig.add_subplot(grid[r, 0]); ax.imshow(R["target_img"]); ax.set_axis_off()
    ax.set_title(f"target\n{R['t']['name']}", fontsize=8.5, color=INK)
    for c, j in enumerate(R["show"]):
        ax = fig.add_subplot(grid[r, 1 + c]); ax.imshow(R["imgs"][c]); ax.set_axis_off()
        head = "best start" if c == 0 else f"start ranked {[1, 2, S // 3 + 1, 2 * S // 3 + 1, S][c]} of {S}"
        ax.set_title(f"{head}\noverlap with target {R['fit'][j]:.2f}\n"
                     + ("" if c == 0 else f"code cosine to best {R['cosine'][j]:.2f}"), fontsize=7.5, color=INK)
    ax = fig.add_subplot(grid[r, 6])
    for j in range(S):
        ax.plot(np.arange(R["history"].shape[0]) * 10, R["history"][:, j], color=LINE, lw=1.0, alpha=0.55)
    ax.set_ylim(0, 1.02); ax.set_xlim(0, steps)
    ax.set_title(f"the {S} descents: overlap with target", fontsize=8.5, color=INK)
    ax.set_xlabel("descent step", fontsize=7.5, color=MUTED)
    bx = fig.add_subplot(grid[r, 7])
    for j in range(S):
        if j != R["best"]:
            bx.plot(np.linspace(0, 1, 11), R["paths"][j, 0], color=LINE, lw=1.0, alpha=0.55)
    bx.set_ylim(0, 1.02); bx.set_xlim(0, 1)
    bx.set_title("straight path from the best code to each\nother end code: overlap with the best fit", fontsize=8.5, color=INK)
    bx.set_xlabel("0 = best code, 1 = other end code", fontsize=7.5, color=MUTED)
    for cx in (ax, bx):
        cx.set_facecolor(PAPER); cx.grid(True, color=GRID, lw=0.6); cx.tick_params(colors=MUTED, labelsize=7, length=0)
        cx.set_yticks([0, 0.5, 1.0])
        for side in cx.spines.values():
            side.set_visible(False)
fig.text(0.035, 0.972, f"{tag} (update {trainer.step_count}): {S} gradient descents on the code toward the same target, from {S} random starts",
         fontsize=13, color=INK, va="center")
fig.text(0.035, 0.948, "Right-hand chart: a line that stays at 1 = a different code giving the same shape with nothing in between changing "
         "(unused directions); a line that dips and returns = two separate codes for one shape; a line that ends low = a different shape.",
         fontsize=9, color=MUTED, va="center")
path = f"{OUT}/data_free_redundancy_paths_{tag}.png"
fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
print("saved", path)
