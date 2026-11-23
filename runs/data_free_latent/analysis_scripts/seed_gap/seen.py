"""Which parts of the surface do one update's 2048 shared points 'see'? (seen location = nearest true surface point)"""
import json, sys
here = __file__
src = open(here.replace("seen.py", "blind.py")).read().split("out = []")[0]
src = src.replace('__file__.replace("blind.py", "compare.py")', repr(here.replace("seen.py", "compare.py")))
exec(src)
from arm_planner.data_free.field import sample_points
cfg = trainer.cfg
axis = torch.linspace(-1, 1, 96, device=dev)
grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
out = []
for k in range(N_SHAPES):
    fn = make_field(k)
    vol = f_only(fn, grid).reshape(96, 96, 96).cpu().numpy()
    if vol.max() <= 0 or vol.min() >= 0: continue
    v, faces, _, _ = measure.marching_cubes(vol, 0.0)
    vt, _, _ = land(fn, torch.tensor(v / 95 * 2 - 1, dtype=torch.float32, device=dev), fixed=2)
    true = vt.cpu().numpy(); tri = true[faces]
    area = np.zeros(len(true)); np.add.at(area, faces.ravel(), np.repeat(0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 3, 3))
    w = area / area.sum()
    hull = ConvexHull(true); depth = np.min(-(true @ hull.equations[:, :3].T + hull.equations[:, 3]), axis=1)
    deep = depth > np.quantile(depth, 0.75)
    tree_true = cKDTree(true)
    def stats_of(P):
        d, _ = cKDTree(P).query(true)
        return dict(cov=float((w * d).sum()), holes=float(w[d > HOLE].sum()), deep=float((w * d)[deep].sum() / w[deep].sum()), n=len(P))
    res = dict(shape=k, deep_area=float(w[deep].sum()))
    for rep in range(3):                                    # three independent updates' worth of points
        x = sample_points(cfg, 1, torch.Generator(device=dev).manual_seed(500 + 31 * k + rep), dev)[0].cpu().numpy()
        x = x[np.linalg.norm(x, axis=1) < float(cfg.bound_radius)]
        _, idx = tree_true.query(x)
        seen = true[np.unique(idx)]
        s = stats_of(seen); s["deep_share"] = float(deep[idx].mean()); s["unique"] = int(len(np.unique(idx))); s["used"] = len(x)
        res.setdefault("uniform", []).append(s)
    gen = torch.Generator(device=dev).manual_seed(1000 + k)
    P, _, _ = blind_near(fn, gen, grow=False); P = P.detach().cpu().numpy()
    sub = P[np.random.default_rng(k).permutation(len(P))[:150]]
    res["F150"] = stats_of(sub); res["F"] = stats_of(P)
    _, iF = tree_true.query(P); res["F"]["deep_share"] = float(deep[iF].mean())
    out.append(res); print(k, flush=True)
json.dump(out, open(sys.argv[3], "w"))
d = out
U = lambda key: np.mean([np.mean([u[key] for u in r["uniform"]]) for r in d])
print(f"shapes {len(d)}; points used per update (|x| < 0.9): {U('used'):.0f}, distinct surface locations seen {U('unique'):.0f}")
print(f"{'':30s} cov     recess  holes")
print(f"{'2048 shared points (seen)':30s} {U('cov'):.4f}  {U('deep'):.4f}  {U('holes'):.3f}")
for m in ('F150', 'F'):
    print(f"{m + ' (n=%.0f)' % np.mean([r[m]['n'] for r in d]):30s} {np.mean([r[m]['cov'] for r in d]):.4f}  {np.mean([r[m]['deep'] for r in d]):.4f}  {np.mean([r[m]['holes'] for r in d]):.3f}")
print(f"most recessed quarter: {np.mean([r['deep_area'] for r in d]):.2f} of the area; share of the 2048 points that see it {U('deep_share'):.3f}; share of F points landing there {np.mean([r['F']['deep_share'] for r in d]):.3f}")
