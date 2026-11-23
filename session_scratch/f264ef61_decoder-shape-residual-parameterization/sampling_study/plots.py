import json, collections, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import LinearSegmentedColormap
base = "/home/eric-luo/.claude/jobs/f264ef61/tmp/sampling_study/J4"
F = "/home/eric-luo/Downloads/pointcloud-motion-planning/figures"
d = json.load(open(f"{base}/results.json"))
SURF, INK, INK2, GRID = "#fcfcfb", "#1f1f1e", "#5f5e58", "#e4e3de"
pal = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "figure.facecolor": SURF, "axes.facecolor": SURF})
# 1. coverage vs budget
tab = collections.defaultdict(list)
for r in d["rows"]:
    tab[(r["scheme"], r["n"])].append(r["cov_mean"])
schemes = list(dict.fromkeys(r["scheme"] for r in d["rows"]))
ns = [256, 512, 1024, 2048, 4096]
fig, ax = plt.subplots(figsize=(9.5, 5.6))
ends = []
for s, c in zip(schemes, pal):
    y = [np.mean(tab[(s, n)]) for n in ns]
    ax.plot(ns, y, color=c, lw=2, marker="o", ms=6, mec=SURF, mew=1.5, label=s)
    ends.append((y[-1], s, c))
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns])
yt = [0.015, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1]
ax.set_yticks(yt); ax.set_yticklabels([f"{v:g}" for v in yt]); ax.minorticks_off()
ax.set_xlabel("sample points per shape (before Newton steps; failed landings dropped)")
ax.set_ylabel("mean distance from the true surface\nto the nearest landed point (lower = better)")
ax.grid(True, which="major", color=GRID, lw=0.8); ax.grid(False, which="minor")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ys = sorted(ends)
placed = []
for y, s, c in ys:                                   # direct labels, nudged apart in log space
    ly = y
    while placed and np.log(ly) - np.log(placed[-1]) < 0.075:
        ly *= np.exp(0.075)
    placed.append(ly)
    ax.annotate(s, (4096, y), xytext=(4096 * 1.12, ly), color=INK, fontsize=9, va="center",
                arrowprops=dict(arrowstyle="-", color=c, lw=1))
ax.set_xlim(220, 4096 * 2.9)
ax.legend(loc="lower left", frameon=False, fontsize=9)
ax.set_title("J4 at 5,000, 32 shapes: how well N samples cover the surface after Newton steps", loc="left", fontsize=11)
fig.tight_layout(); fig.savefig(f"{F}/sampling_coverage_vs_budget.png", dpi=110); plt.close(fig)
# 2. catchment ratio by hull depth (single series)
c = d["catch"]; ratio = np.array([x["ratio"] for x in c]); depth = np.array([x["depth"] for x in c])
bins = [(0, .02, "on the hull\n(depth < 0.02)"), (.02, .06, "shallow\n(0.02-0.06)"), (.06, .12, "recessed\n(0.06-0.12)"), (.12, 9, "deep\n(> 0.12)")]
med = [np.median(ratio[(depth >= lo) & (depth < hi)]) for lo, hi, _ in bins]
fig, ax = plt.subplots(figsize=(7.5, 4.6))
xs = np.arange(len(bins))
ax.bar(xs, med, width=0.6, color=pal[0], edgecolor=SURF, lw=2)
ax.axhline(1, color=INK2, lw=1, ls="--"); ax.text(len(bins) - 0.55, 1.04, "fair share (1)", color=INK2, fontsize=9, ha="right")
for x, m in zip(xs, med):
    ax.text(x, m + 0.04, f"{m:.2f}", ha="center", fontsize=10, color=INK)
ax.set_xticks(xs); ax.set_xticklabels([b[2] for b in bins])
ax.set_ylabel("landing share / area share\n(median over surface patches)")
ax.set_xlabel("how deep the surface patch lies inside the shape's convex hull")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.grid(True, axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.set_title("Uniform sampling over-samples the outside and under-samples recesses", loc="left", fontsize=11)
fig.tight_layout(); fig.savefig(f"{F}/sampling_fairness_by_depth.png", dpi=110); plt.close(fig)
# 3. shapes coloured by sampling ratio (diverging, grey midpoint)
cmap = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#f0efec", "#ef9a8f", "#b3261e"])
fig = plt.figure(figsize=(2.3 * 8, 3.0))
for k in range(8):
    z = np.load(f"{base}/shape{k}.npz")
    v, f, r = z["verts"], z["faces"], z["ratio"]
    face_r = r[f].mean(1)
    t = np.clip(-np.log2(np.clip(face_r, 1e-3, None)) / 2, -1, 1)     # ratio 4 -> -1 (blue), 1 -> 0, 1/4 -> +1 (red)
    col = cmap(0.5 + 0.5 * t)[:, :3]
    tri = v[f]; nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); nrm /= np.linalg.norm(nrm, axis=1, keepdims=True).clip(1e-9)
    light = np.array([0.4, -0.5, 0.75]); light /= np.linalg.norm(light)
    col = col * (0.5 + 0.5 * np.abs(nrm @ light))[:, None]
    ax = fig.add_subplot(1, 8, k + 1, projection="3d"); ax.set_axis_off()
    ax.add_collection3d(Poly3DCollection(tri, facecolors=col, edgecolor="none"))
    ax.set_xlim(-0.6, 0.6); ax.set_ylim(-0.6, 0.6); ax.set_zlim(-0.6, 0.6); ax.set_box_aspect((1, 1, 1)); ax.view_init(22, 35)
    ax.set_title(f"shape {k}", fontsize=9)
fig.suptitle("Where uniform samples land: red = under-sampled (1/4 of its fair share or less), grey = fair, blue = over-sampled (4x or more)", fontsize=11)
fig.tight_layout(); fig.savefig(f"{F}/sampling_fairness_on_shapes.png", dpi=100); plt.close(fig)
print("ok")
