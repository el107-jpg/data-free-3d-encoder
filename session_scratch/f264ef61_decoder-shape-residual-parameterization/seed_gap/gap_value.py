"""How much does gap filling fix?  Targeted gap points vs the same number of untargeted extra points.

usage: gap_value.py decoded <checkpoint> <out.json> | gap_value.py thingi <anchors.pt> <out.json>
"""
import json, sys
src_code = open(__file__.replace("gap_value.py", "compare.py")).read().split("# ------------------------------------------------------------------ evaluation")[0]
exec(src_code)
HOLE = 0.08

def first_stage(variant, fn, gen):
    if variant == "shells":
        x = shell_points(gen, 192)
        p, nn_, ok = land(fn, x)
        d = torch.cdist(x, x); d.fill_diagonal_(float("inf")); nb = d.topk(6, largest=False).indices
        a = torch.arange(192, device=dev)[:, None].expand_as(nb).reshape(-1); b = nb.reshape(-1)
        pairs = (p[a], p[b], ok[a] & ok[b])
        extra = lambda g: land(fn, shell_points(g, 128))
    else:
        pin, _, okin = land(fn, 0.05 * D_IN); pout, _, okout = land(fn, 0.95 * D_OUT)
        p = torch.cat([pin, pout]); ok = torch.cat([okin, okout])
        As, Bs, Os = [], [], []
        for P, o, NB in ((pin, okin, NB_IN), (pout, okout, NB_OUT)):
            a = torch.arange(len(P), device=dev)[:, None].expand_as(NB).reshape(-1)
            As.append(P[a]); Bs.append(P[NB.reshape(-1)]); Os.append(o[a] & o[NB.reshape(-1)])
        a = torch.arange(64, device=dev)[:, None].expand_as(IN_OUT).reshape(-1)
        As.append(pin[a]); Bs.append(pout[IN_OUT.reshape(-1)]); Os.append(okin[a] & okout[IN_OUT.reshape(-1)])
        pairs = (torch.cat(As), torch.cat(Bs), torch.cat(Os))
        # untargeted extra: 128 more seeds in fresh random directions, half inner, half outer
        def extra(g):
            d = torch.nn.functional.normalize(torch.randn(128, 3, generator=g, device=dev), dim=-1)
            return land(fn, torch.cat([0.05 * d[:43], 0.95 * d[43:]]))
    return p[ok], pairs, extra

out = []
axis = torch.linspace(-1, 1, 96, device=dev)
grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
for k in range(N_SHAPES):
    fn = make_field(k)
    vol = f_only(fn, grid).reshape(96, 96, 96).cpu().numpy()
    if vol.max() <= 0 or vol.min() >= 0:
        continue
    v, faces, _, _ = measure.marching_cubes(vol, 0.0)
    vt, _, _ = land(fn, torch.tensor(v / 95 * 2 - 1, dtype=torch.float32, device=dev), fixed=2)
    true = vt.cpu().numpy(); tri = true[faces]
    area = np.zeros(len(true)); np.add.at(area, faces.ravel(), np.repeat(0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 3, 3))
    w = area / area.sum()
    hull = ConvexHull(true); depth = np.min(-(true @ hull.equations[:, :3].T + hull.equations[:, 3]), axis=1)
    deep = depth > np.quantile(depth, 0.75)
    def stats(P):
        d, _ = cKDTree(P).query(true)
        return dict(cov=float((w * d).sum()), holes=float(w[d > HOLE].sum()), deep=float((w * d)[deep].sum() / w[deep].sum()))
    for variant in ("shells", "seeds"):
        gen = torch.Generator(device=dev).manual_seed(1000 + k)
        S, (A, B, OK), extra = first_stage(variant, fn, gen)
        pg, _, okg = fill_gaps(fn, A, B, OK, gen, 128); G = pg[okg]
        pr, _, okr = extra(torch.Generator(device=dev).manual_seed(7000 + k)); Rr = pr[okr]
        Sn, Gn, Rn = S.cpu().numpy(), G.cpu().numpy(), Rr.cpu().numpy()
        tS = cKDTree(Sn)
        hit_g = float((tS.query(Gn)[0] > HOLE).mean()) if len(Gn) else 0.
        hit_r = float((tS.query(Rn)[0] > HOLE).mean()) if len(Rn) else 0.
        # surface newly covered (within HOLE of a new point but not of any first-stage point)
        dS, _ = tS.query(true)
        new_g = float(w[(dS > HOLE) & (cKDTree(Gn).query(true)[0] <= HOLE)].sum()) if len(Gn) else 0.
        new_r = float(w[(dS > HOLE) & (cKDTree(Rn).query(true)[0] <= HOLE)].sum()) if len(Rn) else 0.
        out.append(dict(shape=k, variant=variant, n_first=len(Sn), n_gap=len(Gn), n_rand=len(Rn),
                        before=stats(Sn), gap=stats(np.concatenate([Sn, Gn])), rand=stats(np.concatenate([Sn, Rn])),
                        hit_gap=hit_g, hit_rand=hit_r, new_gap=new_g, new_rand=new_r))
    print("shape", k, flush=True)
json.dump(out, open(outp, "w")); print("saved", outp)
