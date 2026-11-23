import sys
here = __file__
src = open(here.replace("recess_depth.py", "batch_shared.py")).read().split("cfg = trainer.cfg\nout = []")[0]
src = src.replace('here.replace("batch_shared.py", "blind.py")', repr(here.replace("recess_depth.py", "blind.py")))
src = src.replace('here.replace("batch_shared.py", "compare.py")', repr(here.replace("recess_depth.py", "compare.py")))
exec(src)
rows = []
for k in range(N_SHAPES):
    fn = make_field(k)
    vol = f_only(fn, grid).reshape(96, 96, 96).cpu().numpy()
    v, faces, _, _ = measure.marching_cubes(vol, 0.0)
    true = v / 95 * 2 - 1
    hull = ConvexHull(true); depth = np.min(-(true @ hull.equations[:, :3].T + hull.equations[:, 3]), axis=1)
    thr = np.quantile(depth, 0.75); ext = np.ptp(true, axis=0).max()
    tri = true[faces]; V = abs(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6)
    rows.append((k, thr, depth[depth > thr].mean(), depth.max(), ext, V / hull.volume))
r = np.array(rows)
print("per shape: threshold depth (75th pct), mean depth of the recessed quarter, max depth, shape size (largest extent), solidity")
for q in (0, 25, 50, 75, 100):
    print(f"  {q:3d}th pct over shapes: threshold {np.percentile(r[:,1], q):.3f}  recess mean depth {np.percentile(r[:,2], q):.3f}  max {np.percentile(r[:,3], q):.3f}  size {np.percentile(r[:,4], q):.2f}  solidity {np.percentile(r[:,5], q):.2f}")
print("shapes whose recessed quarter starts deeper than 0.05:", int((r[:,1] > 0.05).sum()), "/", len(r), "; shallower than 0.02:", int((r[:,1] < 0.02).sum()))
