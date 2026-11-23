"""Does evaluating the slope (eikonal) and consistency terms at separately sampled points reduce their conflict with
diversity?  usage: grad_resample.py <label>=<checkpoint> [...]
Per pass (12 codes from the trainer's sampler, 8 passes): diversity on the pass's points; eikonal + consistency
(checkpoint multipliers) at (a) the same points, (b) an independent draw from the same distribution, (c) a draw from
the other distribution (uniform <-> shells). Cosines raw and after Adam's per-weight scaling.
"""
import copy, sys, torch
from arm_planner.data_free.train import Trainer
from arm_planner.data_free.field import evaluate_field, sample_points
from arm_planner.data_free import losses as L

def flat(gs, params):
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(gs, params)])
cos = lambda a, b: float(a @ b / (a.norm() * b.norm()).clamp_min(1e-30))
for item in sys.argv[1:]:
    label, path = item.split("=", 1)
    t = Trainer.from_checkpoint(path, device="cuda", tensorboard=False)
    model, cfg = t.model, t.cfg
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    v = [t.optimizer.state[p]["exp_avg_sq"].reshape(-1) if "exp_avg_sq" in t.optimizer.state.get(p, {}) else torch.ones(p.numel(), device=p.device) for p in params]
    pre = 1.0 / (torch.cat(v).sqrt() + 1e-8)
    me, mc = t.multipliers.multiplier("eikonal"), t.multipliers.multiplier("consistency")
    other = copy.copy(cfg); other.point_sampling = "uniform" if cfg.point_sampling == "shells" else "shells"
    gen = torch.Generator(device="cuda").manual_seed(99)
    res = {k: [] for k in ("a_raw", "b_raw", "c_raw", "a_adam", "b_adam", "c_adam", "eik_eik_raw", "eik_eik_adam")}
    for rep in range(8):
        b = t.sample_batch()
        codes, cl = b.codes[:12], b.cluster[:12]
        fb = evaluate_field(model, codes, cl, b.x, cfg)
        div, _ = L.nearest_pool(fb, fb.f.new_zeros((0, fb.f.shape[1] - 1)), torch.arange(1, fb.f.shape[1], device=fb.f.device), cfg)
        div = float(getattr(cfg, "variety_weight", 1.0)) * div
        field_same = me * L.eikonal(fb) + mc * L.consistency(fb)
        gd = flat(torch.autograd.grad(div, params, retain_graph=True, allow_unused=True), params)
        ga = flat(torch.autograd.grad(field_same, params, allow_unused=True), params)
        del fb
        grads = {}
        for key, c in (("b", cfg), ("c", other)):
            x = sample_points(c, 1, gen, "cuda")
            fb2 = evaluate_field(model, codes, cl, x, c)
            grads[key] = flat(torch.autograd.grad(me * L.eikonal(fb2) + mc * L.consistency(fb2), params, allow_unused=True), params)
            del fb2
        for space, s in (("raw", 1.0), ("adam", pre)):
            res[f"a_{space}"].append(cos(ga * s, gd * s))
            res[f"b_{space}"].append(cos(grads["b"] * s, gd * s))
            res[f"c_{space}"].append(cos(grads["c"] * s, gd * s))
            res[f"eik_eik_{space}"].append(cos(ga * s, grads["b"] * s))
    m = lambda k: sum(res[k]) / len(res[k])
    print(f"\n{label} (diversity on {cfg.point_sampling} points)")
    for space in ("raw", "adam"):
        print(f"  {space:4s}: cos(field terms, diversity): same points {m('a_' + space):+.3f}; independent {cfg.point_sampling} draw {m('b_' + space):+.3f}; "
              f"{other.point_sampling} draw {m('c_' + space):+.3f};  field-term gradient, same vs independent points: cos {m('eik_eik_' + space):+.3f}")
