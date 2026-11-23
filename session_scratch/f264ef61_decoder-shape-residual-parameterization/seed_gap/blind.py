"""Certified-ball blind-spot probing vs gap filling and untargeted extra points.

usage: blind.py decoded <checkpoint> <out.json> | blind.py thingi <anchors.pt> <out.json>
Every evaluated point x gives a ball of radius |f(x)|/L with no surface in it (outside if f < 0, solid if f > 0).
Uncertified grid cells, minus those within HOLE of a landed point, form pockets; only pockets touching
certified-inside or known-surface cells (connected to the shape) are probed, largest first.
"""
import json, sys
src_code = open(__file__.replace("blind.py", "compare.py")).read().split("# ------------------------------------------------------------------ evaluation")[0]
exec(src_code)
from scipy import ndimage
HOLE, GRID = 0.08, 32
L = 1.8 if kind == "decoded" else 1.0
LOG = []
LAST = {}
import os
VERSIONS = os.environ.get("VERSIONS", "A,B").split(",")
GAP_MULT = float(os.environ.get("GAP_MULT", 1.0)); TUBE = int(os.environ.get("TUBE", 3)); CONE = float(os.environ.get("CONE", 0))
_f_grad = f_grad
def f_grad(fn, x):                       # record every evaluated position and value (land() calls this)
    f, g = _f_grad(fn, x)
    LOG.append((x.detach(), f.detach()))
    return f, g

cg = torch.linspace(-1 + 1 / GRID, 1 - 1 / GRID, GRID, device=dev)
CELLS = torch.stack(torch.meshgrid(cg, cg, cg, indexing="ij"), -1).reshape(-1, 3)
HALF = (2 / GRID) * 3 ** 0.5 / 2
OUTER = (CELLS.norm(dim=-1) > 0.9) if kind == "decoded" else (CELLS.abs().max(-1).values > 0.95)

def pockets(landed):
    """Candidate pockets: uncertified, not near known surface, connected to the shape. Returns (labels, ids, stats)."""
    X = torch.cat([x for x, _ in LOG]); F = torch.cat([f for _, f in LOG])
    out_c = OUTER.clone(); in_c = torch.zeros_like(out_c)
    r = F.abs() / L - HALF
    keep = r > 0
    X, F, r = X[keep], F[keep], r[keep]
    for s in range(0, len(CELLS), 4096):
        d = torch.cdist(CELLS[s:s + 4096], X)
        inside_ball = d <= r[None]
        out_c[s:s + 4096] |= (inside_ball & (F[None] < 0)).any(1)
        in_c[s:s + 4096] |= (inside_ball & (F[None] > 0)).any(1)
    near = torch.as_tensor(cKDTree(landed).query(CELLS.cpu().numpy())[0] <= HOLE, device=dev)
    unc = ~(out_c | in_c | near)
    U = unc.reshape(GRID, GRID, GRID).cpu().numpy()
    shape = (in_c | near).reshape(GRID, GRID, GRID).cpu().numpy()
    outside = out_c.reshape(GRID, GRID, GRID).cpu().numpy()
    # enclosure: share of 26 grid rays (up to 8 cells) that reach the shape side before open space
    closed = np.zeros(U.shape)
    dirs = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) != (0, 0, 0)]
    for dvec in dirs:
        state = np.zeros(U.shape, dtype=np.int8)          # 0 undecided, 1 closed, 2 open
        for k in range(1, 9):
            sh = tuple(-k * c for c in dvec)
            hit_shape = np.roll(shape, sh, axis=(0, 1, 2)); hit_out = np.roll(outside, sh, axis=(0, 1, 2))
            # cells whose ray leaves the grid at this step are open
            off = np.zeros(U.shape, dtype=bool)
            for ax, c in enumerate(dvec):
                if c > 0: off[(slice(None),) * ax + (slice(GRID - k, None),)] = True
                if c < 0: off[(slice(None),) * ax + (slice(0, k),)] = True
            und = state == 0
            state[und & off] = 2
            und = state == 0
            state[und & hit_shape] = 1
            state[und & ~hit_shape & hit_out] = 2
        closed += state == 1
    enclosure = closed / len(dirs)
    cand = U & (enclosure >= 0.5)
    lab, n = ndimage.label(cand)
    sizes = ndimage.sum(cand, lab, range(1, n + 1)) if n else []
    ids = [i + 1 for i, sz in enumerate(sizes) if sz >= 2]
    LAST["unc"], LAST["cand"] = U, cand
    return lab, ids, dict(uncertified=int(U.sum()), enclosed=int(cand.sum()), pockets=len(ids),
                          pocket_cells=int(sum(sizes[i - 1] for i in ids)) if ids else 0)

def probe(lab, ids, n, gen):
    sizes = np.array([(lab == i).sum() for i in ids], dtype=float)
    if not len(ids):
        return torch.empty(0, 3, device=dev)
    alloc = np.floor(n * sizes / sizes.sum()).astype(int)
    alloc[np.argsort(-sizes)[: n - alloc.sum()]] += 1
    pts = []
    rng = np.random.default_rng(int(torch.randint(0, 10**9, (1,), generator=gen, device=dev)))
    cells = CELLS.cpu().numpy().reshape(GRID, GRID, GRID, 3)
    for i, a in zip(ids, alloc):
        if a <= 0:
            continue
        m = lab == i
        depth = ndimage.distance_transform_edt(m)
        idx = np.argwhere(m)
        first = np.unravel_index(np.argmax(depth), m.shape)          # most enclosed point of the pocket
        chosen = [first] + [tuple(idx[j]) for j in rng.integers(0, len(idx), a - 1)]
        for c in chosen:
            pts.append(cells[c] + rng.uniform(-1, 1, 3) * (1 / GRID))
    return torch.tensor(np.array(pts), dtype=torch.float32, device=dev)

def blind(fn, gen, n_first=192, rounds=(48, 48, 32)):
    LOG.clear()
    p, nn_, ok = land(fn, shell_points(gen, n_first))
    P, N = [p[ok]], [nn_[ok]]
    stats = []
    for n in rounds:
        lab, ids, st = pockets(torch.cat(P).cpu().numpy()); stats.append(st)
        x = probe(lab, ids, n, gen)
        if len(x) == 0:
            break
        q, qn, qok = land(fn, x.clamp(-1, 1))
        P.append(q[qok]); N.append(qn[qok])
    lab, ids, st = pockets(torch.cat(P).cpu().numpy()); stats.append(st)
    return torch.cat(P), torch.cat(N), stats

def gap_weights(starts, landed, okm, k=6):
    """Gap tubes on the grid: pairs of points that started close but landed far apart (score = excess landing
    distance over the median); cells along the segment between the two landings get the score."""
    S = torch.as_tensor(starts, device=dev); P = torch.as_tensor(landed, device=dev); OKm = torch.as_tensor(okm, device=dev)
    d = torch.cdist(S, S); d.fill_diagonal_(float("inf"))
    nb = d.topk(min(k, len(S) - 1), largest=False).indices
    a = torch.arange(len(S), device=dev)[:, None].expand_as(nb).reshape(-1); b = nb.reshape(-1)
    ok = OKm[a] & OKm[b]
    dist = (P[a] - P[b]).norm(dim=-1)
    score = (dist - GAP_MULT * dist[ok].median()).clamp_min(0) * ok
    keep = score > 0
    A, B, sc = P[a][keep], P[b][keep], score[keep]
    if CONE > 0:
        # double cone around each gap segment (apexes at the landings, radius CONE x gap length at the middle,
        # capped at 0.3), with a Gaussian fall-off: weight = score x exp(-d^2 / 2 sigma(t)^2), sigma(t) = cone radius
        Wc = torch.zeros(len(CELLS), device=dev)
        for s0 in range(0, len(A), 128):
            a0, b0, s1 = A[s0:s0 + 128], B[s0:s0 + 128], sc[s0:s0 + 128]
            ab = b0 - a0; L2 = ab.square().sum(-1).clamp_min(1e-9)                # pairs
            rel = CELLS[None] - a0[:, None]                                        # pairs x cells x 3
            t = ((rel * ab[:, None]).sum(-1) / L2[:, None]).clamp(0, 1)
            dperp = (rel - t[..., None] * ab[:, None]).norm(dim=-1)
            rmax = (CONE * L2.sqrt()).clamp(max=0.3)
            sigma = (rmax[:, None] * (1 - (2 * t - 1).abs())).clamp_min(1.0 / GRID)
            Wc += (s1[:, None] * torch.exp(-dperp.square() / (2 * sigma.square()))).sum(0)
        return Wc.reshape(GRID, GRID, GRID).cpu().numpy()
    t = torch.linspace(0.1, 0.9, 9, device=dev)
    pts = A[:, None] + t[None, :, None] * (B - A)[:, None]                       # pairs x 9 x 3
    idx = ((pts + 1) / 2 * GRID).long().clamp(0, GRID - 1)
    flat = (idx[..., 0] * GRID + idx[..., 1]) * GRID + idx[..., 2]
    W = torch.zeros(GRID ** 3, device=dev)
    W.index_add_(0, flat.reshape(-1), sc[:, None].expand(-1, 9).reshape(-1))
    W = W.reshape(GRID, GRID, GRID).cpu().numpy()
    return ndimage.maximum_filter(W, size=TUBE) if TUBE > 1 else W

def outside_estimate(cells_xyz, P, N, k=8):
    """Side of the landed point cloud: vote of the k nearest landings' tangent planes, weighted 1/distance.
    > 0 means estimated outside. Uses only landings and their outward normals (no field evaluations)."""
    X = torch.as_tensor(cells_xyz, dtype=torch.float32, device=dev)
    d = torch.cdist(X, P); kk = min(k, len(P))
    dk, ik = d.topk(kk, largest=False)
    rel = X[:, None] - P[ik]                                              # cells x k x 3
    side = (rel * N[ik]).sum(-1) / rel.norm(dim=-1).clamp_min(1e-6)
    w = 1.0 / (dk + 1e-3)
    return ((side * w).sum(-1) / w.sum(-1)).cpu().numpy()

def blind_out(fn, gen, hard, n_first=192, rounds=(48, 48, 32)):
    """Plain blind-spot probing, but emphasising pocket cells the landed point cloud says are outside the shape."""
    LOG.clear()
    p, nn_, ok = land(fn, shell_points(gen, n_first))
    P, N = [p[ok]], [nn_[ok]]
    rng = np.random.default_rng(int(torch.randint(0, 10**9, (1,), generator=gen, device=dev)))
    cells = CELLS.cpu().numpy().reshape(GRID, GRID, GRID, 3)
    info = []
    for n in rounds:
        Pc, Nc = torch.cat(P), torch.cat(N)
        lab, ids, _ = pockets(Pc.cpu().numpy())
        if not len(ids):
            break
        pk = np.isin(lab, ids); idx_all = np.argwhere(pk)
        est = outside_estimate(cells[pk], Pc, Nc)
        E = np.zeros(lab.shape); E[pk] = est
        outm = pk & (E > 0)
        # diagnostic only: how good is the estimate on these cells
        truth = (f_only(fn, torch.tensor(cells[pk], dtype=torch.float32, device=dev)) <= 0).cpu().numpy(); EVALS[0] -= int(pk.sum())
        acc = float(((est > 0) == truth).mean()); recall_out = float((est[truth] > 0).mean()) if truth.any() else 1.0
        Wt = np.where(outm, 1.0, 0.0 if hard else 0.1) * pk
        if Wt.sum() <= 0:
            Wt = pk.astype(float)                                          # no estimated-outside cells: plain method
        sizes = np.array([Wt[lab == i].sum() for i in ids])
        keep = sizes > 0; ids_k = [i for i, kp in zip(ids, keep) if kp]; sizes = sizes[keep]
        alloc = np.floor(n * sizes / sizes.sum()).astype(int)
        alloc[np.argsort(-sizes)[: n - alloc.sum()]] += 1
        pts = []
        for i, a in zip(ids_k, alloc):
            if a <= 0:
                continue
            m = lab == i
            mo = m & outm
            depth = ndimage.distance_transform_edt(mo if mo.any() else m)
            first = np.unravel_index(np.argmax(depth), m.shape)
            idx = np.argwhere(m); w = Wt[tuple(idx.T)]
            chosen = [first] + [tuple(idx[j]) for j in rng.choice(len(idx), size=a - 1, p=w / w.sum())]
            for c in chosen:
                pts.append(cells[c] + rng.uniform(-1, 1, 3) / GRID)
        q0 = torch.tensor(np.array(pts), dtype=torch.float32, device=dev).clamp(-1, 1)
        info.append(dict(candidates=int(outm.sum()), pocket=int(pk.sum()), acc=acc, recall_out=recall_out,
                         true_out=float(truth.mean()), inside=float((f_only(fn, q0) > 0).float().mean())))
        EVALS[0] -= len(q0)
        q, qn, qok = land(fn, q0)
        P.append(q[qok]); N.append(qn[qok])
    return torch.cat(P), torch.cat(N), info

def surface_estimate(cells_xyz, P, N, k=8):
    """Signed depth below the estimated surface (tangent planes of the k nearest landings, weighted 1/distance;
    > 0 inside) and distance to the nearest landing. Landings and normals only, no field evaluations."""
    X = torch.as_tensor(cells_xyz, dtype=torch.float32, device=dev)
    d = torch.cdist(X, P); dk, ik = d.topk(min(k, len(P)), largest=False)
    h = ((P[ik] - X[:, None]) * N[ik]).sum(-1)
    w = 1.0 / (dk + 1e-3)
    return ((h * w).sum(-1) / w.sum(-1)).cpu().numpy(), dk[:, 0].cpu().numpy()

def blind_near(fn, gen, grow, n_first=192, rounds=(48, 48, 32), sigma0=0.08, floor=0.1):
    """Pocket probing with weight = novelty (distance to nearest landing) x inside falloff
    (1 outside; floor + (1-floor) exp(-h^2 / 2 sigma^2) inside, sigma = sigma0 [+ novelty if grow])."""
    LOG.clear()
    p, nn_, ok = land(fn, shell_points(gen, n_first))
    P, N = [p[ok]], [nn_[ok]]
    rng = np.random.default_rng(int(torch.randint(0, 10**9, (1,), generator=gen, device=dev)))
    cells = CELLS.cpu().numpy().reshape(GRID, GRID, GRID, 3)
    info = []
    for n in rounds:
        Pc, Nc = torch.cat(P), torch.cat(N)
        lab, ids, _ = pockets(Pc.cpu().numpy())
        if not len(ids):
            break
        pk = np.isin(lab, ids)
        h, dn = surface_estimate(cells[pk], Pc, Nc)
        sigma = sigma0 + (dn if grow else 0.0)
        inside_f = np.where(h > 0, floor + (1 - floor) * np.exp(-h ** 2 / (2 * sigma ** 2)), 1.0)
        W = np.zeros(lab.shape); W[pk] = (dn / dn.mean()) * inside_f
        sizes = np.array([W[lab == i].sum() for i in ids])
        alloc = np.floor(n * sizes / sizes.sum()).astype(int)
        alloc[np.argsort(-sizes)[: n - alloc.sum()]] += 1
        pts = []
        for i, a in zip(ids, alloc):
            if a <= 0:
                continue
            idx = np.argwhere(lab == i); w = W[tuple(idx.T)]
            chosen = [int(np.argmax(w))] + list(rng.choice(len(idx), size=a - 1, p=w / w.sum()))
            for j in chosen:
                pts.append(cells[tuple(idx[j])] + rng.uniform(-1, 1, 3) / GRID)
        q0 = torch.tensor(np.array(pts), dtype=torch.float32, device=dev).clamp(-1, 1)
        fq = f_only(fn, q0); EVALS[0] -= len(q0)                            # diagnostics only
        info.append(dict(candidates=int(pk.sum()), inside=float((fq > 0).float().mean()),
                         inside_depth=float(fq[fq > 0].mean()) if (fq > 0).any() else 0.0))
        q, qn, qok = land(fn, q0)
        P.append(q[qok]); N.append(qn[qok])
    return torch.cat(P), torch.cat(N), info

def blind_gap(fn, gen, version, n_first=192, rounds=(48, 48, 32)):
    if version in ("F", "G"):
        return blind_near(fn, gen, grow=(version == "G"), n_first=n_first, rounds=rounds)
    if version in ("D", "E"):
        return blind_out(fn, gen, hard=(version == "E"), n_first=n_first, rounds=rounds)
    """Probe where unproven space overlaps the Newton gap graph. version 'A': enclosed pockets x gaps,
    'B': all unproven x gaps."""
    LOG.clear()
    x = shell_points(gen, n_first)
    p, nn_, ok = land(fn, x)
    starts, lands, oks, P, N = [x], [p], [ok], [p[ok]], [nn_[ok]]
    rng = np.random.default_rng(int(torch.randint(0, 10**9, (1,), generator=gen, device=dev)))
    cells = CELLS.cpu().numpy().reshape(GRID, GRID, GRID, 3)
    info = []
    for n in rounds:
        lab, ids, _ = pockets(torch.cat(P).cpu().numpy())
        if version == "C":
            # plain probing's allocation (enclosed pockets, probes by volume); cone weights only choose
            # where inside each pocket: first probe at the strongest cone cell, the rest by weight + 10% floor
            Wg = gap_weights(torch.cat(starts), torch.cat(lands), torch.cat(oks))
            sizes = np.array([(lab == i).sum() for i in ids], dtype=float)
            if not len(ids):
                break
            alloc = np.floor(n * sizes / sizes.sum()).astype(int)
            alloc[np.argsort(-sizes)[: n - alloc.sum()]] += 1
            pts = []
            for i, a in zip(ids, alloc):
                if a <= 0:
                    continue
                idx = np.argwhere(lab == i)
                w = Wg[tuple(idx.T)]
                w = w + 0.1 * (w.mean() if w.mean() > 0 else 1.0)
                order = [int(np.argmax(w))] + list(rng.choice(len(idx), size=a - 1, p=w / w.sum()))
                for j in order:
                    pts.append(cells[tuple(idx[j])] + rng.uniform(-1, 1, 3) / GRID)
            q0 = torch.tensor(np.array(pts), dtype=torch.float32, device=dev).clamp(-1, 1)
            info.append(dict(candidates=int(np.isin(lab, ids).sum()), inside=float((f_only(fn, q0) > 0).float().mean())))
            EVALS[0] -= len(q0)
            q, qn, qok = land(fn, q0)
            starts.append(q0); lands.append(q); oks.append(qok); P.append(q[qok]); N.append(qn[qok])
            continue
        region = LAST["cand"] if version == "A" else LAST["unc"]
        W = gap_weights(torch.cat(starts), torch.cat(lands), torch.cat(oks)) * region
        if W.sum() <= 0:                                  # nothing overlaps: fall back to the region itself
            W = region.astype(float)
        if W.sum() <= 0:
            break
        flat = W.reshape(-1) / W.sum()
        pick = rng.choice(len(flat), size=n, p=flat)
        q0 = cells.reshape(-1, 3)[pick] + rng.uniform(-1, 1, (n, 3)) / GRID
        q0 = torch.tensor(q0, dtype=torch.float32, device=dev).clamp(-1, 1)
        info.append(dict(candidates=int((W > 0).sum()), inside=float((f_only(fn, q0) > 0).float().mean()) if version else 0))
        EVALS[0] -= len(q0)                                 # that check is diagnostic only, not part of the method
        q, qn, qok = land(fn, q0)
        starts.append(q0); lands.append(q); oks.append(qok); P.append(q[qok]); N.append(qn[qok])
    return torch.cat(P), torch.cat(N), info

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
    def stats_of(P):
        d, _ = cKDTree(P).query(true)
        return dict(cov=float((w * d).sum()), holes=float(w[d > HOLE].sum()), deep=float((w * d)[deep].sum() / w[deep].sum()), p95=float(np.quantile(d, .95)))
    gen = torch.Generator(device=dev).manual_seed(1000 + k)
    EVALS[0] = 0
    P, _, pst = blind(fn, gen)
    ev_blind = EVALS[0]
    Pn = P.detach().cpu().numpy()
    first = Pn[: 192]                                     # approx: first-stage landings come first
    # comparison: shells 320 with stop-when-landed, and shells 192 + 128 gap fill (same first stage seed)
    g2 = torch.Generator(device=dev).manual_seed(1000 + k); EVALS[0] = 0
    s_p, _, s_ok = land(fn, shell_points(g2, 320)); ev_sh = EVALS[0]
    g3 = torch.Generator(device=dev).manual_seed(1000 + k); EVALS[0] = 0
    gp, _ = shells_gap(fn, g3); ev_gap = EVALS[0]
    res_v = {}
    for version in VERSIONS:
        g4 = torch.Generator(device=dev).manual_seed(1000 + k); EVALS[0] = 0
        vp, _, vinfo = blind_gap(fn, g4, version)
        res_v[version] = (vp.detach().cpu().numpy(), EVALS[0], vinfo)
    out.append(dict(shape=k, blind=stats_of(Pn), shells=stats_of(s_p[s_ok].cpu().numpy()), gap=stats_of(gp.detach().cpu().numpy()),
                    **{v: stats_of(res_v[v][0]) for v in res_v},
                    n=dict(blind=len(Pn), shells=int(s_ok.sum()), gap=len(gp), **{v: len(res_v[v][0]) for v in res_v}),
                    evals=dict(blind=ev_blind, shells=ev_sh, gap=ev_gap, **{v: res_v[v][1] for v in res_v}),
                    pockets=pst, **{"info_" + v: res_v[v][2] for v in res_v}))
    print("shape", k, pst[0], "->", pst[-1], flush=True)
json.dump(out, open(outp, "w")); print("saved", outp)
