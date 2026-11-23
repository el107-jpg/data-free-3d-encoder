"""Shared shells + per-shape F gap filling, concatenated across a batch.

usage: N_SHAPES=32 batch_shared.py decoded <checkpoint> <out.json>     (batches of BATCH=16 shapes)
Per batch: 1024 shared shell points (not landed). Per shape: its values there + one Newton step (free in J4 training)
give certified balls and an estimated surface; F weighting then places 64 probes (2 rounds of 32), landed. The 16 x 64
landed gap points join the shells: 2048 shared points. For each shape, the surface each point 'sees' is its nearest
true surface location. Compared with 2048 uniform points (|x| < 0.9 used) and 2048 shared shells.
"""
import json, sys, time
here = __file__
src = open(here.replace("batch_shared.py", "blind.py")).read().split("out = []")[0]
src = src.replace('__file__.replace("blind.py", "compare.py")', repr(here.replace("batch_shared.py", "compare.py")))
exec(src)
from arm_planner.data_free.field import sample_points
import os
BATCH = int(os.environ.get("BATCH", 16)); NSH = 1024; NGAP = 64

def gap_fill(fn, xs, gen, rounds=(32, 32), sigma0=0.08, floor=0.1):
    """F probing, but started from a shape's values at shared, unlanded shell points (+ one free Newton step)."""
    LOG.clear()
    f, g = _f_grad(fn, xs)                                      # the training pass: f and grad at the shells
    gn = g.norm(dim=-1)
    xp = (xs - (f / g.square().sum(-1).clamp_min(1e-8))[:, None] * g).clamp(-1, 1)
    fp = f_only(fn, xp)                                         # J4's field_moves_surface already evaluates this
    LOG.append((xs, f)); LOG.append((xp, fp))
    ok = (fp.abs() < 0.01) & (gn > 0.3)
    P, N = [xp[ok]], [(-g / gn.clamp_min(1e-8)[:, None])[ok]]
    rng = np.random.default_rng(int(torch.randint(0, 10**9, (1,), generator=gen, device=dev)))
    cells = CELLS.cpu().numpy().reshape(GRID, GRID, GRID, 3)
    gaps = []; t_book = 0.0; ev0 = EVALS[0]
    for n in rounds:
        t0 = time.perf_counter()
        Pc, Nc = torch.cat(P), torch.cat(N)
        lab, ids, _ = pockets(Pc.cpu().numpy())
        if not len(ids):
            break
        pk = np.isin(lab, ids)
        h, dn = surface_estimate(cells[pk], Pc, Nc)
        inside_f = np.where(h > 0, floor + (1 - floor) * np.exp(-h ** 2 / (2 * sigma0 ** 2)), 1.0)
        W = np.zeros(lab.shape); W[pk] = (dn / dn.mean()) * inside_f
        sizes = np.array([W[lab == i].sum() for i in ids])
        alloc = np.floor(n * sizes / sizes.sum()).astype(int)
        alloc[np.argsort(-sizes)[: n - alloc.sum()]] += 1
        pts = []
        for i, a in zip(ids, alloc):
            if a <= 0: continue
            idx = np.argwhere(lab == i); w = W[tuple(idx.T)]
            chosen = [int(np.argmax(w))] + list(rng.choice(len(idx), size=a - 1, p=w / w.sum()))
            pts += [cells[tuple(idx[j])] + rng.uniform(-1, 1, 3) / GRID for j in chosen]
        q0 = torch.tensor(np.array(pts), dtype=torch.float32, device=dev).clamp(-1, 1)
        t_book += time.perf_counter() - t0
        q, qn, qok = land(fn, q0)
        P.append(q[qok]); N.append(qn[qok]); gaps.append(q[qok])
    G = torch.cat(gaps) if gaps else torch.empty(0, 3, device=dev)
    if len(G) < NGAP:                                           # pad with extra surface estimates if some probes failed
        extra = P[0][torch.randperm(len(P[0]), generator=torch.Generator().manual_seed(1))[: NGAP - len(G)].to(dev)]
        G = torch.cat([G, extra])
    return G[:NGAP], dict(probe_evals=EVALS[0] - ev0, landed=int(sum(len(p) for p in gaps)), book_s=t_book, shell_landed=int(ok.sum()))

axis = torch.linspace(-1, 1, 96, device=dev)
grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
def truth(fn):
    vol = f_only(fn, grid).reshape(96, 96, 96).cpu().numpy()
    v, faces, _, _ = measure.marching_cubes(vol, 0.0)
    vt, _, _ = land(fn, torch.tensor(v / 95 * 2 - 1, dtype=torch.float32, device=dev), fixed=2)
    true = vt.cpu().numpy(); tri = true[faces]
    area = np.zeros(len(true)); np.add.at(area, faces.ravel(), np.repeat(0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 3, 3))
    hull = ConvexHull(true); depth = np.min(-(true @ hull.equations[:, :3].T + hull.equations[:, 3]), axis=1)
    return true, area / area.sum(), depth > np.quantile(depth, 0.75)

def seen_stats(true, w, deep, X):
    """X: shared points; each sees its nearest true surface location."""
    tree = cKDTree(true); _, idx = tree.query(X)
    S = true[np.unique(idx)]
    d, _ = cKDTree(S).query(true)
    return dict(cov=float((w * d).sum()), holes=float(w[d > HOLE].sum()), deep=float((w * d)[deep].sum() / w[deep].sum()),
                deep_share=float(deep[idx].mean()), unique=int(len(np.unique(idx))))

cfg = trainer.cfg
out = []
for b0 in range(0, N_SHAPES, BATCH):
    ks = list(range(b0, min(b0 + BATCH, N_SHAPES)))
    gb = torch.Generator(device=dev).manual_seed(900 + b0)
    shells_x = shell_points(gb, NSH)                                           # shared by the batch
    shells_2048 = shell_points(torch.Generator(device=dev).manual_seed(950 + b0), 2048)
    uni = sample_points(cfg, 1, torch.Generator(device=dev).manual_seed(970 + b0), dev)[0]
    uni = uni[uni.norm(dim=-1) < float(cfg.bound_radius)]
    gaps, infos = {}, {}
    for k in ks:
        gaps[k], infos[k] = gap_fill(make_field(k), shells_x, torch.Generator(device=dev).manual_seed(1000 + k))
    shared = torch.cat([shells_x] + [gaps[k] for k in ks]).cpu().numpy()
    for k in ks:
        true, w, deep = truth(make_field(k))
        own = torch.cat([shells_x, gaps[k]]).cpu().numpy()
        r = dict(shape=k, info=infos[k],
                 uniform=seen_stats(true, w, deep, uni.cpu().numpy()),
                 shells2048=seen_stats(true, w, deep, shells_2048.cpu().numpy()),
                 shells1024=seen_stats(true, w, deep, shells_x.cpu().numpy()),
                 own_only=seen_stats(true, w, deep, own),
                 proposed=seen_stats(true, w, deep, shared))
        out.append(r); print(k, {m: round(r[m]["holes"], 3) for m in ("uniform", "shells2048", "proposed")}, flush=True)
json.dump(out, open(sys.argv[3], "w"))
M = lambda m, key: np.mean([r[m][key] for r in out])
print(f"\n{len(out)} shapes in batches of {BATCH}; uniform points used (|x| < 0.9): {len(uni)}")
print(f"{'point set (shared by the batch)':44s} cov     recess  holes   recess share of points")
for m, name in (("uniform", "2048 uniform (current)"), ("shells2048", "2048 shared shells"), ("shells1024", "1024 shared shells only"),
                ("own_only", "1024 shells + own 64 gap points"), ("proposed", "1024 shells + 16 x 64 gap points (proposed)")):
    print(f"{name:44s} {M(m,'cov'):.4f}  {M(m,'deep'):.4f}  {M(m,'holes'):.3f}   {M(m,'deep_share'):.3f}")
I = lambda key: np.mean([r['info'][key] for r in out])
print(f"per shape: shells landed by the free Newton step {I('shell_landed'):.0f}/1024, probes landed {I('landed'):.0f}/64, "
      f"probe landing evaluations {I('probe_evals'):.0f}, bookkeeping {1000 * I('book_s'):.0f} ms (CPU, 2 rounds)")
for key in ("cov", "deep", "holes"):
    for a, bm in (("proposed", "uniform"), ("proposed", "shells2048")):
        dd = np.array([r[bm][key] - r[a][key] for r in out])
        print(f"  {a} vs {bm} {key}: {dd.mean():+.4f} +/- {dd.std() / np.sqrt(len(dd)):.4f} (proposed better on {(dd > 0).sum()}/{len(out)})")
