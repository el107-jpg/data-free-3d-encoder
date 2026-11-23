"""Which sample points tell us most about a decoded shape?  (one model, 32 random shapes)

usage: study.py <checkpoint> <out_dir>
Ground truth: marching cubes at RES, vertices refined onto f = 0 with Newton steps; area weights; hull depth; facing.
1. Schemes x budgets: Newton-land the samples (k steps), keep precise landings (|f| < 0.01, |grad| in [0.8, 1.2]),
   coverage = distance from every true surface point to the nearest landing (mean, p95), and a tangent-plane
   reconstruction f_hat(x) = n_i . (p_i - x) from the nearest landing, compared with f near the surface.
2. Catchments: 200k uniform box points -> landings -> nearest true vertex -> per-patch landing share / area share.
3. Landing quality vs where a point started.
"""
import json, sys, math
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree, ConvexHull
from skimage import measure
from arm_planner.data_free.train import Trainer

ckpt, out = sys.argv[1], Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
dev = "cuda"
trainer = Trainer.from_checkpoint(ckpt, device=dev, tensorboard=False)
model = trainer.model.eval()
D = trainer.cfg.latent_dim
import os
SHAPES, RES = int(os.environ.get('SHAPES', 32)), 128
gen = torch.Generator().manual_seed(2024)
codes = torch.nn.functional.normalize(torch.randn(SHAPES, D, generator=gen), dim=-1).to(dev)

def field(mod, x, grad=False):
    """f (and grad f) at x (N x 3), chunked."""
    fs, gs = [], []
    for s in range(0, x.shape[0], 65536):
        xi = x[s:s + 65536][None]
        if grad:
            xi = xi.clone().requires_grad_(True)
            with torch.enable_grad():
                f = model.query(mod, xi)
                (g,) = torch.autograd.grad(f.sum(), xi)
            fs.append(f.detach()[0]); gs.append(g[0])
        else:
            with torch.no_grad():
                fs.append(model.query(mod, xi)[0])
    return (torch.cat(fs), torch.cat(gs)) if grad else torch.cat(fs)

def newton(mod, x, steps):
    for _ in range(steps):
        f, g = field(mod, x, grad=True)
        x = x - (f / g.square().sum(-1).clamp_min(1e-8))[:, None] * g
    f, g = field(mod, x, grad=True)
    return x, f, g

def land(mod, x, steps):
    """Landed points, their normals (outward), and a mask of precise landings."""
    p, f, g = newton(mod, x, steps)
    gn = g.norm(dim=-1)
    ok = (f.abs() < 0.01) & (gn > 0.5) & (p.abs() <= 1).all(-1)
    return p, -g / gn.clamp_min(1e-8)[:, None], ok

def sample(scheme, n, mod, g):
    if scheme.startswith("box"):
        return (torch.rand(n, 3, generator=g) * 2 - 1).to(dev)
    if scheme == "ball":
        d = torch.nn.functional.normalize(torch.randn(n, 3, generator=g), dim=-1)
        return (d * 0.9 * torch.rand(n, 1, generator=g) ** (1 / 3)).to(dev)
    if scheme == "shells":                       # SD2's strata: 5/50/20/25 % over radii 0-.2-.55-.7-.9
        lim = (0, .2, .55, .7, .9); cnt = [round(.05 * n), n // 2, round(.2 * n)]; cnt.append(n - sum(cnt))
        d = torch.nn.functional.normalize(torch.randn(n, 3, generator=g), dim=-1); r = []
        for c, lo, hi in zip(cnt, lim, lim[1:]):
            r.append((lo ** 3 + (hi ** 3 - lo ** 3) * torch.rand(c, generator=g)) ** (1 / 3))
        return (d * torch.cat(r)[:, None]).to(dev)
    pool = (torch.rand(40 * n, 3, generator=g) * 2 - 1).to(dev)
    f = field(mod, pool)
    if scheme == "band":                         # within 0.05 of the surface
        keep = pool[f.abs() < 0.05]
    elif scheme == "inside":
        keep = pool[f > 0]
    return keep[:n] if keep.shape[0] >= n else torch.cat([keep, pool[: n - keep.shape[0]]])

def fps(points, k):
    """Farthest point sampling (best-case even spread), on CPU."""
    pts = points; n = pts.shape[0]; chosen = [0]; d = np.full(n, np.inf)
    for _ in range(k - 1):
        d = np.minimum(d, np.linalg.norm(pts - pts[chosen[-1]], axis=1)); chosen.append(int(d.argmax()))
    return np.array(chosen)

schemes = [("box, 1 step", "box", 1), ("ball, 1 step", "ball", 1), ("SD2 shells, 1 step", "shells", 1),
           ("band |f|<.05, 1 step", "band", 1), ("inside only, 1 step", "inside", 1), ("box, 3 steps", "box3", 3)]
budgets = [256, 512, 1024, 2048, 4096]
rows, catch_rows, start_rows, slope_rows = [], [], [], []
patch_info = {}
for k in range(SHAPES):
    mod = model.prepare(codes[k:k + 1])
    axis = torch.linspace(-1, 1, RES, device=dev)
    grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
    vol = field(mod, grid).reshape(RES, RES, RES).cpu().numpy()
    if vol.max() <= 0 or vol.min() >= 0:
        continue
    v, faces, _, _ = measure.marching_cubes(vol, 0.0)
    v = v / (RES - 1) * 2 - 1
    tri = v[faces]
    area_f = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    area = np.zeros(len(v)); np.add.at(area, faces.ravel(), np.repeat(area_f / 3, 3))
    vt, _, ok_true = land(mod, torch.tensor(v, dtype=torch.float32, device=dev), 2)
    true = vt.cpu().numpy(); w = area / area.sum()
    hull = ConvexHull(true)
    depth = np.min(-(true @ hull.equations[:, :3].T + hull.equations[:, 3]), axis=1)
    _, gt = field(mod, vt, grad=True)
    nrm = (-gt / gt.norm(dim=-1, keepdim=True)).cpu().numpy()
    slope = gt.norm(dim=-1).cpu().numpy()
    slope_rows.append(dict(shape=k, within10=float((w * (np.abs(slope - 1) < 0.1)).sum()),
                           within20=float((w * (np.abs(slope - 1) < 0.2)).sum()), median=float(np.median(slope)),
                           p05=float(np.quantile(slope, .05)), p95=float(np.quantile(slope, .95))))
    facing = (nrm * true).sum(1) / np.linalg.norm(true, axis=1).clip(1e-6)
    tree_true = cKDTree(true)
    # test points near the surface for the reconstruction
    tg = torch.Generator().manual_seed(100 + k)
    test = (torch.rand(400000, 3, generator=tg) * 2 - 1).to(dev)
    ft = field(mod, test); near = ft.abs() < 0.1
    test, ft = test[near][:20000], ft[near][:20000]
    for name, scheme, steps in schemes:
        for n in budgets:
            g = torch.Generator().manual_seed(1000 * k + n)
            x = sample(scheme, n, mod, g)
            p, nn_, ok = land(mod, x, steps)
            p, nn_ = p[ok].cpu().numpy(), nn_[ok].cpu().numpy()
            if len(p) < 4:
                continue
            dcov, _ = cKDTree(p).query(true)
            d_rec, idx = cKDTree(p).query(test.cpu().numpy())
            fhat = ((p[idx] - test.cpu().numpy()) * nn_[idx]).sum(1)
            rec = np.abs(fhat - ft.cpu().numpy())
            rows.append(dict(shape=k, scheme=name, n=n, landed=len(p) / n, cov_mean=float((w * dcov).sum()),
                             cov_p95=float(np.quantile(dcov, .95)), rec_mean=float(rec.mean()), rec_p95=float(np.quantile(rec, .95)),
                             cov_concave=float((w * dcov)[depth > 0.1].sum() / max(w[depth > 0.1].sum(), 1e-9))))
    # best case: farthest-point selection among 20k precise landings
    big, bn, bok = land(mod, sample("box", 20000, mod, torch.Generator().manual_seed(7)), 3)
    bp, bn = big[bok].cpu().numpy(), bn[bok].cpu().numpy()
    order = fps(bp, min(max(budgets), len(bp)))            # prefix of farthest-point order = best n-point spread
    for n in budgets:
        sel = order[:n]
        dcov, _ = cKDTree(bp[sel]).query(true)
        d_rec, idx = cKDTree(bp[sel]).query(test.cpu().numpy())
        fhat = ((bp[sel][idx] - test.cpu().numpy()) * bn[sel][idx]).sum(1)
        rec = np.abs(fhat - ft.cpu().numpy())
        rows.append(dict(shape=k, scheme="even spread (best case)", n=n, landed=1.0, cov_mean=float((w * dcov).sum()),
                         cov_p95=float(np.quantile(dcov, .95)), rec_mean=float(rec.mean()), rec_p95=float(np.quantile(rec, .95)),
                         cov_concave=float((w * dcov)[depth > 0.1].sum() / max(w[depth > 0.1].sum(), 1e-9))))
    # catchments: 200k uniform box points, 3 Newton steps
    cg = torch.Generator().manual_seed(500 + k)
    pool = (torch.rand(200000, 3, generator=cg) * 2 - 1).to(dev)
    f0 = field(mod, pool)
    cp, _, cok = land(mod, pool, 3)
    hits = np.zeros(len(true))
    _, at = tree_true.query(cp[cok].cpu().numpy()); np.add.at(hits, at, 1)
    ratio = (hits / max(hits.sum(), 1)) / w.clip(1e-12)                     # >1 over-sampled, <1 under-sampled
    # 64 patches by k-means on position (+ normal) for a smoother map
    from scipy.cluster.vq import kmeans2
    feat = np.concatenate([true, 0.15 * nrm], 1)
    _, lab = kmeans2(feat, 64, seed=3, minit="++")
    patch_ratio = np.array([(hits[lab == j].sum() / hits.sum()) / max(w[lab == j].sum(), 1e-12) for j in range(64)])
    patch_depth = np.array([(w[lab == j] * depth[lab == j]).sum() / max(w[lab == j].sum(), 1e-12) for j in range(64)])
    patch_face = np.array([(w[lab == j] * facing[lab == j]).sum() / max(w[lab == j].sum(), 1e-12) for j in range(64)])
    patch_area = np.array([w[lab == j].sum() for j in range(64)])
    for j in range(64):
        catch_rows.append(dict(shape=k, ratio=float(patch_ratio[j]), depth=float(patch_depth[j]), facing=float(patch_face[j]), area=float(patch_area[j])))
    if k < 8:
        np.savez(out / f"shape{k}.npz", verts=true, faces=faces, ratio=patch_ratio[lab], depth=depth, label=lab)
    # where good landings start: by start |f| band and inside/outside, 1 step and 3 steps
    f0n = f0.cpu().numpy(); r0 = pool.norm(dim=-1).cpu().numpy()
    p1, _, ok1 = land(mod, pool, 1); ok1 = ok1.cpu().numpy(); ok3 = cok.cpu().numpy()
    for lo, hi in ((0, .02), (.02, .05), (.05, .1), (.1, .2), (.2, .4), (.4, 2)):
        for side, m in (("inside", f0n > 0), ("outside", f0n <= 0)):
            sel = m & (np.abs(f0n) >= lo) & (np.abs(f0n) < hi)
            if sel.sum() == 0:
                continue
            start_rows.append(dict(shape=k, band=f"{lo}-{hi}", side=side, share=float(sel.mean()),
                                   land1=float(ok1[sel].mean()), land3=float(ok3[sel].mean())))
    print(f"shape {k} done", flush=True)
json.dump(dict(rows=rows, catch=catch_rows, start=start_rows, slope=slope_rows), open(out / "results.json", "w"))
print("saved", out / "results.json")
