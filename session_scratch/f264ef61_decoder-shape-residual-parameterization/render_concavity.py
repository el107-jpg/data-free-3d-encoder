"""Render shapes coloured by mean curvature (red = concave, blue = convex) for one model.

usage: render_concavity.py <label> <checkpoint> <out_prefix>
Writes <out_prefix>_random.png (8 fixed random codes) and <out_prefix>_concave.png (the 8 lowest-solidity of 64).
"""
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
from skimage import measure
from arm_planner.data_free.train import Trainer

label, path, out = sys.argv[1:4]
trainer = Trainer.from_checkpoint(path, device="cuda", tensorboard=False)
model = trainer.model.eval()
gen = torch.Generator().manual_seed(2024)
codes = torch.nn.functional.normalize(torch.randn(64, trainer.cfg.latent_dim, generator=gen), dim=-1).cuda()
RES = 80
axis = torch.linspace(-1, 1, RES, device="cuda")
grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(1, -1, 3)

def mesh(k):
    c = codes[k:k + 1]
    with torch.no_grad():
        mod = model.prepare(c)
        f = model.query(mod, grid).reshape(RES, RES, RES).cpu().numpy()
    if f.max() <= 0 or f.min() >= 0:
        return None
    v, fa, _, _ = measure.marching_cubes(f, 0.0)
    v = v / (RES - 1) * 2 - 1
    tri = v[fa]
    vol = abs(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)
    hull = ConvexHull(v)
    sol = vol / hull.volume
    x = torch.tensor(tri.mean(1), dtype=torch.float32, device="cuda")[None].requires_grad_(True)
    val = model.query(mod, x)
    (g,) = torch.autograd.grad(val.sum(), x, create_graph=True)
    n = -g / g.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    div = 0
    for i in range(3):
        (gi,) = torch.autograd.grad(n[..., i].sum(), x, retain_graph=True)
        div = div + gi[..., i]
    centres = tri.mean(1)
    depth = np.min(-(centres @ hull.equations[:, :3].T + hull.equations[:, 3]), axis=1)   # distance inside the hull
    return tri, depth, n.detach().cpu().numpy()[0], sol

def draw(ax, item, title):
    ax.set_axis_off()
    if item is None:
        return
    tri, depth, normal, sol = item
    cmap = plt.get_cmap("coolwarm")
    colour = cmap(np.clip(depth / 0.15, 0, 1))[:, :3]      # blue on the hull, red 0.15 or more inside it
    light = np.array([0.4, -0.5, 0.75]); light /= np.linalg.norm(light)
    shade = 0.45 + 0.55 * np.clip(normal @ light, 0, 1)
    colour = colour * shade[:, None]
    ax.add_collection3d(Poly3DCollection(tri, facecolors=colour, edgecolor="none", linewidths=0))
    ax.set_xlim(-0.6, 0.6); ax.set_ylim(-0.6, 0.6); ax.set_zlim(-0.6, 0.6)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=22, azim=35)
    ax.set_title(f"{title}solidity {sol:.2f}", fontsize=8, pad=-2)

items = [mesh(k) for k in range(64)]
sol = np.array([it[3] if it else 9 for it in items])
for name, pick in (("random", list(range(8))), ("concave", list(np.argsort(sol)[:8]))):
    fig = plt.figure(figsize=(2.2 * 8, 2.5))
    for j, k in enumerate(pick):
        draw(fig.add_subplot(1, 8, j + 1, projection="3d"), items[k], f"code {k}: ")
    fig.suptitle(f"{label}: " + ("8 fixed random codes" if name == "random" else "8 most concave of 64 codes (lowest solidity)")
                 + "  (colour = depth inside the convex hull: blue on the hull, red >= 0.15 inside)", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{out}_{name}.png", dpi=95)
    plt.close(fig)
print("done", label)
