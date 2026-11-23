"""Render thin-wall sites found by the point graph on a saved model.

For each chosen site: the wall-thickness direction is found by probing the field around the
site midpoint, the true thickness is measured along that direction, and the field is sliced
in a plane containing that direction (overview and zoom).
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from arm_planner.data_free import graph as G
from arm_planner.data_free import graph_losses as GL
from arm_planner.data_free import losses as L
from arm_planner.data_free.field import evaluate_field, sample_points
from arm_planner.data_free.train import Trainer

ckpt, out, label = sys.argv[1], sys.argv[2], sys.argv[3]
SHOW, SCAN, CHUNK = 8, 128, 8
trainer = Trainer.from_checkpoint(ckpt, device="cuda", tensorboard=False)
cfg, model = trainer.cfg, trainer.model.eval()
t = float(cfg.min_feature)
gen = torch.Generator().manual_seed(123)

sites = []          # (code index, mid xyz, graph length)
per_shape_counts = []
for c0 in range(0, SCAN, CHUNK):
    codes = trainer.bank.codes[c0:c0 + CHUNK]
    cluster = torch.arange(codes.shape[0], device="cuda")
    x = sample_points(cfg, codes.shape[0], gen, "cuda")
    fb = evaluate_field(model, codes, cluster, x, cfg, create_graph=False)
    e = L.consistency_error(fb)
    with torch.no_grad():
        gr = G.build_graph(fb, e, cfg, mode="full")
        gr = GL.confirm(model, fb.mod if fb.mod is not None else model.prepare(codes), gr, cfg)
    counts = torch.bincount(gr.wall_shape, minlength=codes.shape[0]).tolist()
    per_shape_counts += counts
    for s, m, l in zip(gr.wall_shape.tolist(), gr.wall_mid.tolist(), gr.wall_len.tolist()):
        sites.append((c0 + s, m, l))
    del fb, gr
print(label, "codes scanned", SCAN, "codes with wall sites", sum(c > 0 for c in per_shape_counts),
      "total sites", len(sites), "mean per code", sum(per_shape_counts) / SCAN)
if not sites:
    json.dump({"label": label, "sites": 0}, open(out + ".json", "w"))
    sys.exit(0)


def field(code_index, pts):
    with torch.no_grad():
        return model(trainer.bank.codes[code_index:code_index + 1], pts.reshape(1, -1, 3)).reshape(pts.shape[:-1])


def analyse(code_index, mid):
    mid = torch.tensor(mid, device="cuda")
    d = torch.randn(4096, 3, device="cuda", generator=torch.Generator("cuda").manual_seed(0))
    d = d / d.norm(dim=-1, keepdim=True)
    # thickness direction: the line through the site along which the field falls fastest on both sides
    r = 0.6 * t
    v = field(code_index, mid + r * d) + field(code_index, mid - r * d)
    u = d[v.argmin()]
    s = torch.linspace(-0.3, 0.3, 1201, device="cuda")
    line = field(code_index, mid + s[:, None] * u)
    fm = float(field(code_index, mid[None])[0])
    thickness = float("nan")
    if fm > 0:
        centre = 600
        neg = (line <= 0)
        right = neg[centre:].nonzero()
        left = neg[:centre + 1].flip(0).nonzero()
        if right.numel() and left.numel():
            thickness = float(s[centre + right[0, 0]] - s[centre - left[0, 0]])
    return mid, u, fm, thickness


# choose sites from distinct codes, spread over graph lengths
by_code = {}
for c, m, l in sites:
    by_code.setdefault(c, []).append((l, m))
chosen = []
for c in sorted(by_code, key=lambda k: -len(by_code[k]))[:SHOW]:
    lst = sorted(by_code[c])
    chosen.append((c, lst[len(lst) // 2][1], lst[len(lst) // 2][0], len(lst)))

# measured thickness of ALL sites (for the summary), then the pictures
all_thick, all_fm = [], []
for c, m, l in sites[:600]:
    _, _, fm, th = analyse(c, m)
    all_fm.append(fm); all_thick.append(th)
th = np.array(all_thick); fmv = np.array(all_fm)
summary = {"label": label, "sites": len(sites), "analysed": len(th), "min_feature": t,
           "codes_with_sites": int(sum(c > 0 for c in per_shape_counts)), "codes_scanned": SCAN,
           "midpoint_inside_fraction": float((fmv > 0).mean()),
           "thickness_median": float(np.nanmedian(th)), "thickness_p10": float(np.nanpercentile(th, 10)),
           "thickness_p90": float(np.nanpercentile(th, 90)),
           "fraction_thinner_than_t": float(np.nanmean(th < t)),
           "fraction_unmeasured": float(np.isnan(th).mean())}
print(json.dumps(summary, indent=1))

RES = 300
import os
from matplotlib.colors import LinearSegmentedColormap
FIELD = os.environ.get("FIELD") == "1"
MARK = "#111111" if FIELD else "#d7301f"
# diverging: red = free space (negative), neutral grey at zero, blue = inside (positive)
CMAP = LinearSegmentedColormap.from_list("sdf", ["#a3241c", "#e48b7a", "#f0efec", "#7fa6d6", "#1f4f9c"])
images = {}
fig, axes = plt.subplots(2, len(chosen), figsize=(2.9 * len(chosen), 6.4), squeeze=False)
details = []
for k, (c, m, l, n) in enumerate(chosen):
    mid, u, fm, thick = analyse(c, m)
    w = torch.linalg.cross(u, torch.tensor([0.0, 0.0, 1.0], device="cuda"))
    if w.norm() < 1e-3:
        w = torch.linalg.cross(u, torch.tensor([1.0, 0.0, 0.0], device="cuda"))
    w = w / w.norm()
    # plane through the site spanned by u (thickness direction) and w; overview is centred on the
    # projection of the origin so the whole shape is visible
    o = mid - (mid @ u) * u - (mid @ w) * w          # plane point closest to the origin
    for row, (half, centre) in enumerate([(1.0, o), (0.2, mid)]):
        a = torch.linspace(-half, half, RES, device="cuda")
        A, B = torch.meshgrid(a, a, indexing="xy")
        pts = centre + A[..., None] * u + B[..., None] * w
        f = field(c, pts).cpu().numpy()
        ax = axes[row, k]
        an = a.cpu().numpy()
        if FIELD:
            lim, stepc = ((0.5, 0.1), (0.15, 0.02))[row]
            im = ax.imshow(f, extent=[-half, half, -half, half], origin="lower", cmap=CMAP,
                           vmin=-lim, vmax=lim, interpolation="bilinear")
            levels = [v for v in np.arange(-2.0, 2.0 + 1e-9, stepc) if abs(v) > 1e-9]
            ax.contour(an, an, f, levels=levels, colors="#55555a", linewidths=0.35, linestyles="solid")
            ax.contour(an, an, f, levels=[0], colors="#111111", linewidths=1.8)
            images[row] = im
        else:
            ax.contourf(an, an, f, levels=[0, 10], colors=["#9ecae1"])
            ax.contour(an, an, f, levels=[0], colors="#08519c", linewidths=1.6)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        if row == 0:
            mu, mw = float((mid - o) @ u), float((mid - o) @ w)
            ax.plot([mu], [mw], marker="o", ms=5, color=MARK)
            ax.add_patch(plt.Rectangle((mu - 0.2, mw - 0.2), 0.4, 0.4, fill=False, ec=MARK, lw=1))
            ax.set_title(f"code {c}: {n} sites", fontsize=9)
        else:
            ax.plot([0], [0], marker="o", ms=6, color=MARK)
            ax.plot([-t / 2, t / 2], [-0.17, -0.17], color="k", lw=2)
            ax.text(0, -0.155, f"minimum feature {t:g}", ha="center", va="bottom", fontsize=7)
            txt = f"measured {thick:.3f}" if thick == thick else "midpoint not inside"
            ax.set_title(f"{txt}\ngraph said {l:.3f}", fontsize=9)
    details.append({"code": c, "sites_on_code": n, "graph_len": l, "measured": thick, "f_mid": fm})
axes[0, 0].set_ylabel("whole shape, slice through a site", fontsize=9)
axes[1, 0].set_ylabel("zoom (0.4 wide), thickness direction horizontal", fontsize=9)
if FIELD:
    fig.suptitle(f"Field values on the thin-wall slices: {label}\n"
                 "Blue = inside (positive), red = free space (negative), thick black line = surface (zero); "
                 "thin lines every 0.1 (top) and 0.02 (bottom); black dot = site midpoint", fontsize=11)
    fig.tight_layout(rect=[0, 0, 0.94, 1])
    for row in (0, 1):
        box = axes[row, -1].get_position()
        cax = fig.add_axes([0.95, box.y0, 0.008, box.height])
        cb = fig.colorbar(images[row], cax=cax, extend="both")
        cb.set_label("field value", fontsize=8); cb.ax.tick_params(labelsize=7)
else:
    fig.suptitle(f"Thin-wall sites flagged by the point graph: {label}\n"
                 "Blue: shape; red dot: site midpoint; red box in the top row is the zoomed region", fontsize=11)
    fig.tight_layout()
fig.savefig(out, dpi=100)
summary["shown"] = details
json.dump(summary, open(out + ".json", "w"), indent=1)
print("saved", out)
