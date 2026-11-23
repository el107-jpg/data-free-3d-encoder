"""Does a surface (Chamfer-style) distance on the chosen neighbour pairs change J4's diversity gradient?

usage: N_SHAPES=64 surf_div_grad.py decoded <J4 checkpoint> <out.json>
Each code gets 150 surface points from F probing (base shells). Neighbours are chosen by the plain field distance
on the shared 2048 points (as in training); each chosen pair's distance becomes d_field + w * d_surf, with d_surf
the pose-corrected mean |f_i - f_j| over both shapes' surface points. Exact gradients via dL/dd then chunked backward.
"""
import json, sys
here = __file__
src = open(here.replace("surf_div_grad.py", "blind.py")).read().split("out = []")[0]
src = src.replace('__file__.replace("blind.py", "compare.py")', repr(here.replace("surf_div_grad.py", "compare.py")))
exec(src)
from arm_planner.data_free import losses as LS
from arm_planner.data_free.field import evaluate_field, sample_points, field_and_grad
cfg = trainer.cfg
B = N_SHAPES; K = int(cfg.nearest_k); POW = float(cfg.nearest_power); SLOPE = float(cfg.diversity_slope)
EPS = float(getattr(cfg, "repel_eps", 1e-3)); FLOOR = LS._TARGET_FLOOR; NS = 150

# ---- surface points per code (F probing on base shells)
import os
cache = here.replace("surf_div_grad.py", f"surf_points_{B}.pt")
S = torch.load(cache) if os.path.exists(cache) else []
for k in range(0 if len(S) == 0 else B, B):
    gen = torch.Generator(device=dev).manual_seed(1000 + k)
    P, _, _ = blind_near(make_field(k), gen, grow=False)
    sel = torch.randperm(len(P), generator=torch.Generator().manual_seed(k))[:NS]
    S.append(P[sel].detach())
if not os.path.exists(cache): torch.save(S, cache)
print("surface points per code:", [len(s) for s in S][:8], "...", flush=True)
S = torch.stack([s if len(s) == NS else torch.cat([s, s[:NS - len(s)]]) for s in S])          # B x NS x 3

params = [p for p in model.parameters()]; names = [n for n, _ in model.named_parameters()]
def flat_grad():
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in params]).clone()
def groups(gv):
    out, o = {}, 0
    for n, p in zip(names, params):
        key = "side network + FiLM" if n.startswith(("mapping", "film")) else ("final layer" if n.startswith("out") else "main network hidden")
        out.setdefault(key, []).append(gv[o:o + p.numel()]); o += p.numel()
    return {k: torch.cat(v) for k, v in out.items()}

# ---- field part on the shared 2048 points (as training)
x = sample_points(cfg, 1, torch.Generator(device=dev).manual_seed(77), dev)
cluster = torch.zeros(B, dtype=torch.long, device=dev)
fb = evaluate_field(model, codes, cluster, x, cfg, create_graph=False)
f = (fb.f if getattr(fb, "f_move", None) is None else fb.f_move)
g = fb.g.detach(); X = fb.x.detach()
mask = (X.square().sum(-1) < float(cfg.bound_radius) ** 2).to(f.dtype)
with torch.no_grad():
    fd = f.detach()
    raw = ((fd[:, None] - fd[None]).abs() * mask[:, None]).sum(-1) / mask[:, None].sum(-1)
    raw.fill_diagonal_(float("inf"))
    chosen = raw.topk(K, dim=1, largest=False).indices                                        # B x K
other = f[chosen]
delta = (f[:, None] - other).reshape(B * K, -1)
pm = mask[:, None].expand(-1, K, -1).reshape(B * K, -1)
delta = LS._remove_pose(delta, X[:, None].expand(-1, K, -1, -1).reshape(B * K, -1, 3), g[:, None].expand(-1, K, -1, -1).reshape(B * K, -1, 3),
                        (0.5 * (f[:, None] + other)).detach().reshape(B * K, -1), pm)
d_field = ((delta.abs() * pm).sum(1) / pm.sum(1)).reshape(B, K)                                # in graph

# reference: the trainer's own nearest_pool loss value (no pool) must match ours at w = 0
ref, _ = LS.nearest_pool(fb, f.new_zeros((0, f.shape[1])), torch.arange(f.shape[1], device=dev), cfg)

# ---- surface part per chosen pair: points S_i u S_j, both codes evaluated there (f in graph, g detached)
I = torch.arange(B, device=dev)[:, None].expand(-1, K).reshape(-1); J = chosen.reshape(-1)
def surf_dist(rows):
    Xp = torch.cat([S[I[rows]], S[J[rows]]], dim=1)                                           # r x 2NS x 3
    fi, gi = field_and_grad(model, model.prepare(codes[I[rows]]), Xp, create_graph=False)
    fj, _ = field_and_grad(model, model.prepare(codes[J[rows]]), Xp, create_graph=False)
    dl = LS._remove_pose(fi - fj, Xp, gi.detach(), (0.5 * (fi + fj)).detach(), torch.ones_like(fi))
    return dl.abs().mean(1)
CH = 32
with torch.no_grad():
    d_surf_val = torch.cat([surf_dist(torch.arange(s, min(s + CH, B * K), device=dev)) for s in range(0, B * K, CH)]).reshape(B, K)

def loss_of(d):
    soft = (d + FLOOR).pow(-POW).sum(1).pow(-1 / POW)
    return (-torch.log(soft / SLOPE + EPS)).mean()

def gradient(w):
    """Full gradient for distance d_field + w d_surf; also its field part and surface part separately."""
    df = d_field.detach().clone().requires_grad_(True); ds = d_surf_val.clone().requires_grad_(True)
    lval = loss_of(df + w * ds); cf, cs = torch.autograd.grad(lval, [df, ds])
    model.zero_grad(set_to_none=True)
    (cf * d_field).sum().backward(retain_graph=True); gf = flat_grad()
    model.zero_grad(set_to_none=True)
    if w > 0:
        csf = cs.reshape(-1)
        for s in range(0, B * K, CH):
            rows = torch.arange(s, min(s + CH, B * K), device=dev)
            (csf[rows] * surf_dist(rows)).sum().backward()
    gs = flat_grad()
    return float(lval), gf, gs

cos = lambda a, b: float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-30))
res = dict(ref_loss=float(ref), field_mean=float(d_field.mean()), surf_mean=float(d_surf_val.mean()),
           field_nearest=float(d_field.min(1).values.mean()), surf_nearest=float(d_surf_val.min(1).values.mean()))
# neighbour agreement: surface distance for every pair (no grad), ranked
with torch.no_grad():
    allp = []
    for i in range(B):
        row = []
        for s0 in range(0, B, 16):
            js = torch.arange(s0, min(s0 + 16, B), device=dev); rows_i = torch.full_like(js, i)
            Xp = torch.cat([S[rows_i], S[js]], dim=1)
            fi, gi = field_and_grad(model, model.prepare(codes[rows_i]), Xp, create_graph=False)
            fj = model.query(model.prepare(codes[js]), Xp)
            dl = LS._remove_pose((fi - fj).detach(), Xp, gi.detach(), (0.5 * (fi + fj)).detach(), torch.ones_like(fj))
            row.append(dl.abs().mean(1)); del fi, gi, fj
        allp.append(torch.cat(row))
    SD = torch.stack(allp); SD.fill_diagonal_(float("inf"))
    s_chosen = SD.topk(K, dim=1, largest=False).indices
    overlap = np.mean([len(set(a.tolist()) & set(b.tolist())) / K for a, b in zip(chosen, s_chosen)])
    first_same = float((chosen[:, 0] == s_chosen[:, 0]).float().mean())
    iu = torch.triu_indices(B, B, 1)
    rf = raw[iu[0], iu[1]].cpu().numpy(); rs = SD[iu[0], iu[1]].cpu().numpy()
    from scipy.stats import spearmanr
    res.update(neighbour_overlap=float(overlap), nearest_same=first_same, spearman_all_pairs=float(spearmanr(rf, rs).correlation))
l0, g0f, _ = gradient(0.0)
res["our_loss_w0"] = l0
print(f"loss check: trainer nearest_pool {res['ref_loss']:.4f}, ours (w=0) {l0:.4f}", flush=True)
print(f"distances over chosen pairs: field mean {res['field_mean']:.4f} (nearest {res['field_nearest']:.4f}), "
      f"surface mean {res['surf_mean']:.4f} (nearest {res['surf_nearest']:.4f})")
print(f"neighbours: top-{K} overlap field vs surface ranking {overlap:.2f}, same nearest {first_same:.2f}, spearman over all pairs {res['spearman_all_pairs']:.2f}")
wbal = res["field_mean"] / res["surf_mean"]
res["runs"] = {}
for w in (0.5 * wbal, wbal, 2 * wbal):
    lw, gf, gs = gradient(w); gw = gf + gs
    gr0, grw, grs = groups(g0f), groups(gw), groups(gs)
    r = dict(w=w, loss=lw, size_ratio=float(gw.norm() / g0f.norm()), cos_full=cos(gw, g0f), surf_share=float(gs.norm() / gw.norm()),
             cos_surf_vs_field=cos(gs, g0f), groups={k: dict(cos=cos(grw[k], gr0[k]), surf_share=float(grs[k].norm() / grw[k].norm().clamp_min(1e-30)))
                                                      for k in gr0})
    res["runs"][f"{w:.3f}"] = r
    print(f"w {w:.3f} (surface weight so that w*d_surf ~ {w / wbal:.1f} x d_field): grad size x{r['size_ratio']:.2f}, cos with current {r['cos_full']:.3f}, "
          f"surface part {r['surf_share']:.2f} of full, cos(surface part, current) {r['cos_surf_vs_field']:.3f}")
    for k, v in r["groups"].items():
        print(f"    {k:22s} cos {v['cos']:.3f}  surface part / full {v['surf_share']:.2f}")
json.dump(res, open(sys.argv[3], "w"), indent=1)
