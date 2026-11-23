"""2-D picture of carving: rounded box minus a disc, min form and smooth min, vs the true distance."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

n = 1200
s = np.linspace(-1, 1, n); h = s[1] - s[0]
X, Y = np.meshgrid(s, s)

def rounded_box(x, y, bx, by, r):          # positive inside
    qx, qy = np.abs(x) - bx, np.abs(y) - by
    out = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0))
    return r - (out + np.minimum(np.maximum(qx, qy), 0))

base = rounded_box(X, Y, 0.45, 0.12, 0.08)          # a slab (a plate seen edge-on)
cutter = 0.22 - np.hypot(X - 0.0, Y - 0.25)         # a disc, positive inside
hard = np.minimum(base, -cutter)                    # base minus cutter
k = 0.03
smooth = -k * np.logaddexp(-base / k, cutter / k)   # smooth min(base, -cutter)

inside = hard > 0
true = np.where(inside, distance_transform_edt(inside), -distance_transform_edt(~inside)) * h
err = np.abs(hard) - np.abs(true)                   # <= 0: the min form underestimates
gy, gx = np.gradient(smooth, h)
slope = np.hypot(gx, gy)

fig, ax = plt.subplots(1, 4, figsize=(18, 4.8))
lv = np.arange(-0.6, 0.61, 0.05)
for a, f, title in ((ax[0], base, "base: rounded box (slab)"),
                    (ax[1], hard, "min(base, -cutter): the cut slab")):
    a.contourf(X, Y, f, levels=lv, cmap="RdBu", extend="both")
    a.contour(X, Y, f, levels=[0], colors="k", linewidths=2)
    a.contour(X, Y, cutter, levels=[0], colors="#d08000", linewidths=1.5, linestyles="--")
    a.set_title(title)
im = ax[2].imshow(-err, origin="lower", extent=(-1, 1, -1, 1), cmap="magma", vmin=0, vmax=0.15)
ax[2].contour(X, Y, hard, levels=[0], colors="w", linewidths=1.5)
ax[2].set_title("how far |f| falls short of the true distance")
fig.colorbar(im, ax=ax[2], fraction=0.046)
im = ax[3].imshow(slope, origin="lower", extent=(-1, 1, -1, 1), cmap="viridis", vmin=0.6, vmax=1.05)
ax[3].contour(X, Y, smooth, levels=[0], colors="w", linewidths=1.5)
ax[3].set_title(f"slope of the smooth min (k = {k})")
fig.colorbar(im, ax=ax[3], fraction=0.046)
for a in ax:
    a.set_aspect("equal"); a.set_xlim(-0.75, 0.75); a.set_ylim(-0.5, 0.65)
ax[0].text(-0.72, 0.58, "orange dashes: cutter surface", color="#d08000", fontsize=9)
fig.tight_layout()
out = "/home/eric-luo/Downloads/pointcloud-motion-planning/figures/cutter_carving_sdf_2d.png"
fig.savefig(out, dpi=110)
print(out)
band = np.abs(true) < 0.02
print("max shortfall anywhere %.3f, within 0.02 of the surface %.4f" % (-err.min(), -err[band].min()))
print("share of the box where the shortfall > 0.01: %.3f" % (( -err > 0.01).mean()))
print("smooth min: smallest slope %.3f; zero set moved by up to about %.3f" % (slope.min(), k*np.log(2)))
