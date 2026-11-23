"""How close is each model's field to the true signed distance of its own surface, by distance from the surface?

usage: sdf_truth.py <label>=<checkpoint> [...]
32 random codes (the same for every model). Surface: marching cubes at 160^3, vertices refined onto f = 0 with two
Newton steps. 20,000 points uniform in the ball of radius 0.9 per code; true distance = distance to the nearest surface
vertex, signed by the field. Per band of true distance: mean |f - d| (value error), mean ||grad f| - 1| (slope error),
and the one-step Newton miss: distance from x - f g/|g|^2 to the surface.
"""
import sys
import numpy as np
import torch
from scipy.spatial import cKDTree
from skimage import measure
from arm_planner.data_free.train import Trainer
from arm_planner.data_free.field import field_and_grad

RES, NPTS = 160, 20000
BANDS = [(0, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 2.0)]
codes0 = torch.nn.functional.normalize(torch.randn(32, 64, generator=torch.Generator().manual_seed(31)), dim=-1)
for item in sys.argv[1:]:
    label, path = item.split("=", 1)
    t = Trainer.from_checkpoint(path, device="cuda", tensorboard=False)
    model = t.model.eval()
    ax = torch.linspace(-1, 1, RES, device="cuda")
    grid = torch.stack(torch.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3)
    rows = {b: [] for b in BANDS}
    for k in range(32):
        mod = model.prepare(codes0[k:k + 1].cuda())
        q = lambda x: model.query(mod, x[None])[0]
        with torch.no_grad():
            vol = torch.cat([q(grid[s:s + 262144]) for s in range(0, len(grid), 262144)]).reshape(RES, RES, RES).cpu().numpy()
        if vol.max() <= 0 or vol.min() >= 0: continue
        v, _, _, _ = measure.marching_cubes(vol, 0.0)
        p = torch.tensor(v / (RES - 1) * 2 - 1, dtype=torch.float32, device="cuda")
        for _ in range(2):
            f, g = field_and_grad(model, mod, p[None], create_graph=False)
            p = (p - (f[0] / g[0].square().sum(-1).clamp_min(1e-8))[:, None] * g[0]).detach()
        tree = cKDTree(p.cpu().numpy())
        gen = torch.Generator().manual_seed(100 + k)
        x = torch.randn(NPTS, 3, generator=gen); x = x / x.norm(dim=-1, keepdim=True) * 0.9 * torch.rand(NPTS, 1, generator=gen) ** (1 / 3)
        x = x.cuda()
        f, g = field_and_grad(model, mod, x[None], create_graph=False)
        f, g = f[0].detach(), g[0].detach()
        d, _ = tree.query(x.cpu().numpy())
        d = torch.tensor(d, device="cuda", dtype=torch.float32)
        sd = torch.where(f >= 0, d, -d)
        land = x - (f / g.square().sum(-1).clamp_min(1e-8))[:, None] * g
        miss, _ = tree.query(land.cpu().numpy())
        miss = torch.tensor(miss, device="cuda", dtype=torch.float32)
        verr = (f - sd).abs(); serr = (g.norm(dim=-1) - 1).abs()
        for lo, hi in BANDS:
            m = (d >= lo) & (d < hi)
            if m.any(): rows[(lo, hi)].append((float(verr[m].mean()), float(serr[m].mean()), float(miss[m].mean()), float((verr[m] / d[m].clamp_min(0.01)).mean()), int(m.sum())))
    print(f"\n{label}: band of true distance | value error |f - d| | relative |f-d|/d | slope error ||g|-1| | one-step Newton miss  (codes {len(rows[BANDS[0]])})")
    for b in BANDS:
        r = np.array(rows[b])
        print(f"   {b[0]:.2f}-{b[1]:.2f}: {r[:, 0].mean():.4f}   {r[:, 3].mean():.3f}   {r[:, 1].mean():.4f}   {r[:, 2].mean():.4f}")
