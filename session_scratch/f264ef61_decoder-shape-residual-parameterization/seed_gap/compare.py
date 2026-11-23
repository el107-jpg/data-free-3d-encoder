"""Seed-and-gap-fill sampling vs other schemes, on decoded shapes or Thingi10K SDF grids.

usage: compare.py decoded <checkpoint> <out.json>   |   compare.py thingi <anchors.pt> <out.json>
Per shape (320 landed points per method): coverage of the true surface (mean, p95, in recesses), tangent-plane
reconstruction error near the surface, field evaluations used and wall time. Plus the shared point set: the union
of all shapes' points, judged per shape.
"""
import json, math, sys, time
import numpy as np
import torch
from scipy.spatial import cKDTree, ConvexHull
from skimage import measure

kind, src, outp = sys.argv[1:4]
dev = "cuda"
import os
N_SHAPES = int(os.environ.get("N_SHAPES", 32))
torch.manual_seed(0)

# ------------------------------------------------------------------ shape sources
if kind == "decoded":
    from arm_planner.data_free.train import Trainer
    trainer = Trainer.from_checkpoint(src, device=dev, tensorboard=False)
    model = trainer.model.eval()
    codes = torch.nn.functional.normalize(torch.randn(N_SHAPES, trainer.cfg.latent_dim, generator=torch.Generator().manual_seed(2024)), dim=-1).to(dev)
    def make_field(k):
        mod = model.prepare(codes[k:k + 1])
        return lambda x: model.query(mod, x[None])[0]
else:
    from arm_planner.data_free.losses import sample_grid
    data = torch.load(src, map_location="cpu", weights_only=False)
    grids = data["grid"][:N_SHAPES].float().to(dev)
    if not data.get("positive_inside", True):
        grids = -grids
    def make_field(k):
        g = grids[k:k + 1]
        return lambda x: sample_grid(g, x.clamp(-1, 1)[None])[0]

EVALS = [0]
def f_only(fn, x):
    out = []
    with torch.no_grad():
        for s in range(0, len(x), 65536):
            out.append(fn(x[s:s + 65536]))
    EVALS[0] += len(x)
    return torch.cat(out)

def f_grad(fn, x):
    fs, gs = [], []
    for s in range(0, len(x), 65536):
        xi = x[s:s + 65536].detach().requires_grad_(True)
        with torch.enable_grad():
            f = fn(xi)
            (g,) = torch.autograd.grad(f.sum(), xi)
        fs.append(f.detach()); gs.append(g)
    EVALS[0] += len(x)
    return torch.cat(fs), torch.cat(gs)

def land(fn, x, max_steps=6, fixed=None):
    """Newton steps until |f| < 0.01 (or `fixed` steps). Returns landings, outward normals, success mask."""
    x = x.clone()
    active = torch.ones(len(x), dtype=torch.bool, device=dev)
    for step in range(fixed if fixed is not None else max_steps):
        idx = active.nonzero()[:, 0] if fixed is None else torch.arange(len(x), device=dev)
        if len(idx) == 0:
            break
        f, g = f_grad(fn, x[idx])
        x[idx] = (x[idx] - (f / g.square().sum(-1).clamp_min(1e-8))[:, None] * g).clamp(-1, 1)
        if fixed is None:
            f2 = f_only(fn, x[idx]); EVALS[0] -= len(idx)        # the convergence check is free in practice (next step's f)
            active[idx] = f2.abs() >= 0.01
    f, g = f_grad(fn, x); EVALS[0] -= len(x)                      # final check, also reused for normals
    gn = g.norm(dim=-1)
    ok = (f.abs() < 0.01) & (gn > 0.3)
    return x, -g / gn.clamp_min(1e-8)[:, None], ok

def fib(n):
    i = torch.arange(n, dtype=torch.float64) + 0.5
    phi = torch.acos(1 - 2 * i / n); th = math.pi * (1 + 5 ** 0.5) * i
    return torch.stack([torch.cos(th) * torch.sin(phi), torch.sin(th) * torch.sin(phi), torch.cos(phi)], -1).float()

D_IN, D_OUT = fib(64).to(dev), fib(128).to(dev)
def neighbours(dirs, k):
    sim = dirs @ dirs.T; sim.fill_diagonal_(-2)
    return sim.topk(k, dim=1).indices
NB_IN, NB_OUT = neighbours(D_IN, 6), neighbours(D_OUT, 6)
IN_OUT = (D_IN @ D_OUT.T).topk(2, dim=1).indices

def seed_gap(fn, gen, n_gap=128):
    pin, nin, okin = land(fn, 0.05 * D_IN)
    pout, nout, okout = land(fn, 0.95 * D_OUT)
    pairs = []
    for P, ok, NB in ((pin, okin, NB_IN), (pout, okout, NB_OUT)):
        a = torch.arange(len(P), device=dev)[:, None].expand_as(NB)
        pairs.append((P[a.reshape(-1)], P[NB.reshape(-1)], (ok[a.reshape(-1)] & ok[NB.reshape(-1)])))
    a = torch.arange(64, device=dev)[:, None].expand_as(IN_OUT)
    pairs.append((pin[a.reshape(-1)], pout[IN_OUT.reshape(-1)], okin[a.reshape(-1)] & okout[IN_OUT.reshape(-1)]))
    A = torch.cat([p[0] for p in pairs]); B = torch.cat([p[1] for p in pairs]); OK = torch.cat([p[2] for p in pairs])
    dist = (A - B).norm(dim=-1)
    s0 = dist[OK].median() if OK.any() else torch.tensor(0., device=dev)
    score = ((dist - s0).clamp_min(0) * OK)
    if float(score.sum()) <= 0:
        score = OK.float()
    pick = torch.multinomial(score / score.sum(), n_gap, replacement=True, generator=gen)
    t = (0.5 + 0.2 * torch.randn(n_gap, 1, generator=gen, device=dev)).clamp(0, 1)
    seg = B[pick] - A[pick]
    x = A[pick] + t * seg + 0.15 * seg.norm(dim=-1, keepdim=True) * torch.randn(n_gap, 3, generator=gen, device=dev)
    pg, ng, okg = land(fn, x.clamp(-1, 1))
    P = torch.cat([pin, pout, pg]); Nn = torch.cat([nin, nout, ng]); OKa = torch.cat([okin, okout, okg])
    return P[OKa], Nn[OKa]

def fill_gaps(fn, A, B, OK, gen, n_gap):
    """Gap filling shared by the methods: pairs (A, B) of landings that should be near each other but are
    far apart get points along the segment (Gaussian around the middle, scattered), then landed."""
    dist = (A - B).norm(dim=-1)
    s0 = dist[OK].median() if OK.any() else torch.tensor(0., device=dev)
    score = (dist - s0).clamp_min(0) * OK
    if float(score.sum()) <= 0:
        score = OK.float()
    pick = torch.multinomial(score / score.sum(), n_gap, replacement=True, generator=gen)
    t = (0.5 + 0.2 * torch.randn(n_gap, 1, generator=gen, device=dev)).clamp(0, 1)
    seg = B[pick] - A[pick]
    x = A[pick] + t * seg + 0.15 * seg.norm(dim=-1, keepdim=True) * torch.randn(n_gap, 3, generator=gen, device=dev)
    return land(fn, x.clamp(-1, 1))

SHELL_MIXES = {"base": (.05, .50, .20), "core": (.20, .50, .15), "inner": (.15, .60, .10)}
def shell_points(gen, n):
    import os
    mix = os.environ.get("SHELL_MIX", "base")
    d = torch.nn.functional.normalize(torch.randn(n, 3, generator=gen, device=dev), dim=-1)
    if mix == "radial":                                   # uniform in radius over [0, 0.9]
        return d * (0.9 * torch.rand(n, generator=gen, device=dev))[:, None]
    fr = SHELL_MIXES[mix]
    lim = (0, .2, .55, .7, .9); cnt = [round(f * n) for f in fr]; cnt.append(n - sum(cnt)); r = []
    for c, lo, hi in zip(cnt, lim, lim[1:]):
        r.append((lo ** 3 + (hi ** 3 - lo ** 3) * torch.rand(c, generator=gen, device=dev)) ** (1 / 3))
    return d * torch.cat(r)[:, None]

def shells_gap(fn, gen, n_seed=192, n_gap=128, k=6):
    """SD2 shells first (landed until they land), then gap filling: each point is paired with its k nearest
    by STARTING position; a pair that starts close but lands far apart marks a gap."""
    x = shell_points(gen, n_seed)
    p, nn_, ok = land(fn, x)
    d = torch.cdist(x, x); d.fill_diagonal_(float("inf"))
    nb = d.topk(k, largest=False).indices
    a = torch.arange(n_seed, device=dev)[:, None].expand_as(nb).reshape(-1); b = nb.reshape(-1)
    pg, ng, okg = fill_gaps(fn, p[a], p[b], ok[a] & ok[b], gen, n_gap)
    P = torch.cat([p, pg]); Nn = torch.cat([nn_, ng]); OKa = torch.cat([ok, okg])
    return P[OKa], Nn[OKa]

def uniform(fn, gen, n=320, steps=3):
    p, nn_, ok = land(fn, torch.rand(n, 3, generator=gen, device=dev) * 2 - 1, fixed=steps)
    return p[ok], nn_[ok]

def shells(fn, gen, n=320, steps=3):
    lim = (0, .2, .55, .7, .9); cnt = [round(.05 * n), n // 2, round(.2 * n)]; cnt.append(n - sum(cnt))
    d = torch.nn.functional.normalize(torch.randn(n, 3, generator=gen, device=dev), dim=-1); r = []
    for c, lo, hi in zip(cnt, lim, lim[1:]):
        r.append((lo ** 3 + (hi ** 3 - lo ** 3) * torch.rand(c, generator=gen, device=dev)) ** (1 / 3))
    p, nn_, ok = land(fn, d * torch.cat(r)[:, None], fixed=steps)
    return p[ok], nn_[ok]

def band(fn, gen, n=320):
    pool = torch.rand(40 * n, 3, generator=gen, device=dev) * 2 - 1
    f = f_only(fn, pool)
    keep = pool[f.abs() < 0.05][:n]
    p, nn_, ok = land(fn, keep, fixed=1)
    return p[ok], nn_[ok]

def even(fn, gen, n=320):
    p, nn_, ok = land(fn, torch.rand(20000, 3, generator=gen, device=dev) * 2 - 1, fixed=3)
    p, nn_ = p[ok].cpu().numpy(), nn_[ok].cpu().numpy()
    chosen = [0]; dmin = np.full(len(p), np.inf)
    for _ in range(min(n, len(p)) - 1):
        dmin = np.minimum(dmin, np.linalg.norm(p - p[chosen[-1]], axis=1)); chosen.append(int(dmin.argmax()))
    return torch.tensor(p[chosen], device=dev), torch.tensor(nn_[chosen], device=dev)

METHODS = [("SD2 shells + gap fill", shells_gap), ("seed + gap fill (yours)", seed_gap), ("uniform box, 3 steps", uniform), ("SD2 shells, 3 steps", shells),
           ("near-surface band", band), ("even spread (best case, costly)", even)]

# ------------------------------------------------------------------ evaluation
RES = 96
rows, unions, shapes = [], {m: [] for m, _ in METHODS}, []
axis = torch.linspace(-1, 1, RES, device=dev)
grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
for k in range(N_SHAPES):
    fn = make_field(k)
    vol = f_only(fn, grid).reshape(RES, RES, RES).cpu().numpy()
    if vol.max() <= 0 or vol.min() >= 0:
        continue
    v, faces, _, _ = measure.marching_cubes(vol, 0.0)
    v = torch.tensor(v / (RES - 1) * 2 - 1, dtype=torch.float32, device=dev)
    vt, _, _ = land(fn, v, fixed=2)
    true = vt.cpu().numpy(); tri = true[faces]
    area = np.zeros(len(true)); np.add.at(area, faces.ravel(), np.repeat(0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 3, 3))
    w = area / area.sum()
    hull = ConvexHull(true); depth = np.min(-(true @ hull.equations[:, :3].T + hull.equations[:, 3]), axis=1)
    deep = depth > np.quantile(depth, 0.75)                      # the most recessed quarter of the surface
    tg = torch.Generator(device=dev).manual_seed(100 + k)
    test = torch.rand(300000, 3, generator=tg, device=dev) * 2 - 1
    ft = f_only(fn, test); near = ft.abs() < 0.1; test, ft = test[near][:20000].cpu().numpy(), ft[near][:20000].cpu().numpy()
    shapes.append(dict(fn=fn, true=true, w=w))
    for name, method in METHODS:
        gen = torch.Generator(device=dev).manual_seed(1000 + k)
        torch.cuda.synchronize(); EVALS[0] = 0; t0 = time.perf_counter()
        p, nn_ = method(fn, gen)
        torch.cuda.synchronize(); ms = (time.perf_counter() - t0) * 1000; evals = EVALS[0]
        p, nn_ = p.detach().cpu().numpy(), nn_.detach().cpu().numpy()
        unions[name].append(p)
        dcov, _ = cKDTree(p).query(true)
        _, idx = cKDTree(p).query(test)
        rec = np.abs(((p[idx] - test) * nn_[idx]).sum(1) - ft)
        rows.append(dict(shape=k, method=name, landed=len(p), evals=evals, ms=ms, cov=float((w * dcov).sum()),
                         cov_p95=float(np.quantile(dcov, .95)), cov_deep=float((w * dcov)[deep].sum() / w[deep].sum()),
                         rec=float(rec.mean())))
    print("shape", k, flush=True)
# shared sets: the union of all shapes' points, judged per shape (points within 0.02 of its surface)
union_rows = []
for name in unions:
    U = torch.tensor(np.concatenate(unions[name]), dtype=torch.float32, device=dev)
    for s in shapes:
        f = f_only(s["fn"], U).abs().cpu().numpy()
        near = U.cpu().numpy()[f < 0.02]
        dcov = cKDTree(near).query(s["true"])[0] if len(near) else np.full(len(s["true"]), 9.)
        union_rows.append(dict(method=name, size=len(U), near_share=float((f < 0.05).mean()), cov=float((s["w"] * dcov).sum())))
json.dump(dict(rows=rows, union=union_rows), open(outp, "w"))
print("saved", outp)
