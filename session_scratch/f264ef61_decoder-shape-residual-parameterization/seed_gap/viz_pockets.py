"""Picture: where the enclosed blind-spot pockets are, against what the filter rejected (decoded shapes)."""
import sys
src = open(__file__.replace("viz_pockets.py", "blind.py")).read().split("out = []")[0]
src = src.replace('__file__.replace("blind.py", "compare.py")', repr(__file__.replace("viz_pockets.py", "compare.py")))
exec(src)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
_pockets = pockets
captured = {}
def pockets_keep(landed):                      # also keep the masks for the picture
    lab, ids, st = _pockets(landed)
    captured["lab"], captured["ids"] = lab, ids
    return lab, ids, st
fig = plt.figure(figsize=(4.2 * 4, 4.6))
axis96 = torch.linspace(-1, 1, 96, device=dev)
g96 = torch.stack(torch.meshgrid(axis96, axis96, axis96, indexing="ij"), -1).reshape(-1, 3)
for j, k in enumerate((0, 1, 2, 5)):
    fn = make_field(k)
    LOG.clear()
    gen = torch.Generator(device=dev).manual_seed(1000 + k)
    p, _, ok = land(fn, shell_points(gen, 192)); landed = p[ok].cpu().numpy()
    # recompute masks for the picture
    X = torch.cat([x for x, _ in LOG]); F = torch.cat([f for _, f in LOG])
    lab, ids, st = pockets(landed)
    cells = CELLS.cpu().numpy().reshape(GRID, GRID, GRID, 3)
    pocket = np.isin(lab, ids)
    # rejected = uncertified but not enclosed: rebuild uncertified mask the same way as pockets()
    out_c = OUTER.clone(); in_c = torch.zeros_like(out_c); r = F.abs() / L - HALF; keep = r > 0
    Xk, Fk, rk = X[keep], F[keep], r[keep]
    for s0 in range(0, len(CELLS), 4096):
        d = torch.cdist(CELLS[s0:s0 + 4096], Xk); inb = d <= rk[None]
        out_c[s0:s0 + 4096] |= (inb & (Fk[None] < 0)).any(1); in_c[s0:s0 + 4096] |= (inb & (Fk[None] > 0)).any(1)
    near = torch.as_tensor(cKDTree(landed).query(CELLS.cpu().numpy())[0] <= HOLE, device=dev)
    unc = (~(out_c | in_c | near)).reshape(GRID, GRID, GRID).cpu().numpy()
    rejected = unc & ~pocket
    vol = f_only(fn, g96).reshape(96, 96, 96).cpu().numpy()
    v, faces, _, _ = measure.marching_cubes(vol, 0.0); v = v / 95 * 2 - 1
    ax = fig.add_subplot(1, 4, j + 1, projection="3d"); ax.set_axis_off()
    ax.add_collection3d(Poly3DCollection(v[faces][::3], facecolors=(0.45, 0.6, 0.85, 0.12), edgecolor="none"))
    rj = cells[rejected][::4]
    ax.scatter(rj[:, 0], rj[:, 1], rj[:, 2], s=1, c="#b5b4ad", alpha=0.25, depthshade=False)
    pk = cells[pocket]
    ax.scatter(pk[:, 0], pk[:, 1], pk[:, 2], s=9, c="#d0342c", alpha=0.9, depthshade=False)
    ax.scatter(landed[:, 0], landed[:, 1], landed[:, 2], s=3, c="#1f1f1e", depthshade=False)
    ax.set_xlim(-0.75, 0.75); ax.set_ylim(-0.75, 0.75); ax.set_zlim(-0.75, 0.75); ax.set_box_aspect((1, 1, 1)); ax.view_init(22, 35)
    ax.set_title(f"shape {k}: {st['pockets']} pockets, {int(pocket.sum())} cells\n(rejected unproven: {int(rejected.sum())} cells)", fontsize=9)
fig.suptitle("Blind-spot pockets after 192 shell points (J4 decoded shapes): red = enclosed pockets kept, grey = unproven space rejected as outside, black = landed points", fontsize=10)
fig.tight_layout()
out = "/home/eric-luo/Downloads/pointcloud-motion-planning/figures/blind_spot_pockets_J4.png"
fig.savefig(out, dpi=100); print(out)
