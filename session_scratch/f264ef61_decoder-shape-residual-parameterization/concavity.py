"""Concavity of the shapes at 64 random codes (the same codes for every model).

usage: concavity.py <label>=<checkpoint> [...]
Per shape: solidity = volume / convex-hull volume (1 = convex); concave share = share of surface points with
mean curvature clearly negative (div of the outward normal < -1, i.e. bending inward with radius < ~2);
inward share = share of surface points whose outward normal faces the centre (cos < 0).
"""
import sys
import numpy as np
import torch
from scipy.spatial import ConvexHull
from skimage import measure
from arm_planner.data_free.train import Trainer

RES = 96
for item in sys.argv[1:]:
    label, path = item.split("=", 1)
    trainer = Trainer.from_checkpoint(path, device="cuda", tensorboard=False)
    model = trainer.model.eval()
    gen = torch.Generator().manual_seed(2024)
    codes = torch.nn.functional.normalize(torch.randn(64, trainer.cfg.latent_dim, generator=gen), dim=-1).cuda()
    axis = torch.linspace(-1, 1, RES, device="cuda")
    grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(1, -1, 3)
    solidity, concave, inward, flat = [], [], [], []
    for k in range(64):
        c = codes[k:k + 1]
        with torch.no_grad():
            mod = model.prepare(c)
            f = torch.cat([model.query(mod, grid[:, s:s + 200000]) for s in range(0, grid.shape[1], 200000)], 1)
        f = f.reshape(RES, RES, RES).cpu().numpy()
        if f.max() <= 0 or f.min() >= 0:
            continue
        verts, faces, _, _ = measure.marching_cubes(f, 0.0)
        verts = verts / (RES - 1) * 2 - 1
        tri = verts[faces]
        volume = abs(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)
        try:
            hull = ConvexHull(verts).volume
            solidity.append(volume / hull)
        except Exception:
            pass
        # surface points: face centres, area weights
        centres = torch.tensor(tri.mean(1), dtype=torch.float32, device="cuda")[None]
        area = torch.tensor(0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1),
                            device="cuda")
        divs = []
        for s in range(0, centres.shape[1], 20000):
            x = centres[:, s:s + 20000].clone().requires_grad_(True)
            val = model.query(mod, x)
            (g,) = torch.autograd.grad(val.sum(), x, create_graph=True)
            n_out = -g / g.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            div = 0
            for i in range(3):
                (gi,) = torch.autograd.grad(n_out[..., i].sum(), x, retain_graph=True)
                div = div + gi[..., i]
            facing = (n_out * x).sum(-1) / x.norm(dim=-1).clamp_min(1e-6)
            divs.append(torch.stack([div.detach(), facing.detach()], -1))
        d = torch.cat(divs, 1)[0]
        w = area / area.sum()
        concave.append(float((w * (d[:, 0] < -1.0)).sum()))
        flat.append(float((w * (d[:, 0].abs() <= 1.0)).sum()))
        inward.append(float((w * (d[:, 1] < 0)).sum()))
    s, cc, iw, fl = map(np.array, (solidity, concave, inward, flat))
    se = lambda a: a.std() / np.sqrt(len(a))
    print(f"{label} (update {trainer.step_count}, {len(s)} shapes): solidity mean {s.mean():.3f} +/- {se(s):.3f} (lowest tenth {np.quantile(s, .1):.3f}); "
          f"concave surface {cc.mean():.3f} +/- {se(cc):.3f}; inward-facing {iw.mean():.3f} +/- {se(iw):.3f}; nearly flat {fl.mean():.3f}", flush=True)
