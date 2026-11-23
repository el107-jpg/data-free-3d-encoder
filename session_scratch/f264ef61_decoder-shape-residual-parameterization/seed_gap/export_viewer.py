"""Export shapes + blind-spot layers for the interactive viewer.  usage: export_viewer.py decoded|thingi <src> <out.json> <ids>"""
import sys, json
here = __file__
src = open(here.replace("export_viewer.py", "blind.py")).read().split("out = []")[0]
src = src.replace('__file__.replace("blind.py", "compare.py")', repr(here.replace("export_viewer.py", "compare.py")))
exec(src)
want = [int(v) for v in sys.argv[4].split(",")]
r3 = lambda a: np.round(np.asarray(a, dtype=float), 3).tolist()
shapes = []
axis64 = torch.linspace(-1, 1, 64, device=dev)
g64 = torch.stack(torch.meshgrid(axis64, axis64, axis64, indexing="ij"), -1).reshape(-1, 3)
if want == [-1]:                                     # pick the two least solid of the first 32
    sol = []
    for k in range(N_SHAPES):
        fn = make_field(k); vol = f_only(fn, g64).reshape(64, 64, 64).cpu().numpy()
        if vol.max() <= 0 or vol.min() >= 0: continue
        v, fa, _, _ = measure.marching_cubes(vol, 0.0); v = v / 63 * 2 - 1; t = v[fa]
        V = abs(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6)
        sol.append((V / ConvexHull(v).volume, k))
    want = [k for _, k in sorted(sol)[:2]]
for k in want:
    fn = make_field(k)
    vol = f_only(fn, g64).reshape(64, 64, 64).cpu().numpy()
    v, fa, _, _ = measure.marching_cubes(vol, 0.0); v = v / 63 * 2 - 1
    LOG.clear()
    gen = torch.Generator(device=dev).manual_seed(1000 + k)
    p, _, ok = land(fn, shell_points(gen, 192)); landed = p[ok].cpu().numpy()
    X = torch.cat([x for x, _ in LOG]).cpu().numpy(); F = torch.cat([f for _, f in LOG]).cpu().numpy()
    lab, ids, st = pockets(landed)
    cells = CELLS.cpu().numpy().reshape(GRID, GRID, GRID, 3)
    pk = np.isin(lab, ids); unc = LAST["unc"]
    rejected = cells[unc & ~pk]
    pocket_cells = cells[pk]
    truth = f_only(fn, torch.tensor(pocket_cells, dtype=torch.float32, device=dev)).cpu().numpy() if len(pocket_cells) else np.zeros(0)
    # probing rounds, as the plain method does them
    P = [p[ok]]; probes = []
    for rnd, n in enumerate((48, 48, 32)):
        lab_r, ids_r, _ = pockets(torch.cat(P).cpu().numpy())
        x = probe(lab_r, ids_r, n, gen)
        if len(x) == 0: break
        q, _, qok = land(fn, x.clamp(-1, 1))
        P.append(q[qok])
        for a, b, o in zip(x.cpu().numpy(), q.cpu().numpy(), qok.cpu().numpy()):
            probes.append(dict(s=r3(a), e=r3(b), ok=int(o), r=rnd))
    rad = np.abs(F) / L
    keep = rad > 0.03
    rng = np.random.default_rng(k)
    rej = rejected[rng.permutation(len(rejected))[:2500]]
    shapes.append(dict(name=f"{'decoded shape' if kind == 'decoded' else 'Thingi10K object'} {k}" + ("" if kind == "decoded" else f" (#{data['file_ids'][k]})"),
                       kind=kind, verts=r3(v.reshape(-1)), faces=fa.reshape(-1).tolist(), shell=r3(landed.reshape(-1)),
                       balls=[[*r3(xx), round(float(rr), 3), int(ff > 0)] for xx, rr, ff in zip(X[keep], rad[keep], F[keep])],
                       rejected=r3(rej.reshape(-1)), pockets=r3(pocket_cells.reshape(-1)), pocket_inside=(truth > 0).astype(int).tolist(),
                       probes=probes, stats=dict(unproven=int(unc.sum()), rejected=int((unc & ~pk).sum()), pocket_cells=int(pk.sum()),
                                                 pockets=len(ids), inside_share=round(float((truth > 0).mean()), 2) if len(truth) else 0, L=L)))
    print("exported", k, st, flush=True)
json.dump(shapes, open(sys.argv[3], "w"), separators=(",", ":"))
print("saved", sys.argv[3])
