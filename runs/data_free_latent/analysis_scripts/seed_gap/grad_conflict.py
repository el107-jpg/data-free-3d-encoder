"""How much do the slope (eikonal) and consistency gradients oppose the diversity gradient? (what PCGrad would remove)

usage: grad_conflict.py <label>=<checkpoint> [...]
Per checkpoint, 6 training-like passes of 12 codes (the trainer's own batch sampler, points and multipliers);
diversity = the nearest-neighbour term among the pass's 16 codes. Gradients over all decoder weights, raw and
after Adam's per-weight scaling (the checkpoint's second-moment estimates), which is what the step follows.
"""
import sys, torch
from arm_planner.data_free.train import Trainer
from arm_planner.data_free.field import evaluate_field
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
    v = []
    for p in params:
        st = t.optimizer.state.get(p, {})
        v.append(st["exp_avg_sq"].reshape(-1) if "exp_avg_sq" in st else torch.ones(p.numel(), device=p.device))
    pre = 1.0 / (torch.cat(v).sqrt() + 1e-8)
    me, mc = t.multipliers.multiplier("eikonal"), t.multipliers.multiplier("consistency")
    rows = []
    for rep in range(6):
        b = t.sample_batch()
        fb = evaluate_field(model, b.codes[:12], b.cluster[:12], b.x, cfg)
        div, _ = L.nearest_pool(fb, fb.f.new_zeros((0, fb.f.shape[1] - 1)), torch.arange(1, fb.f.shape[1], device=fb.f.device), cfg)
        div = float(getattr(cfg, "variety_weight", 1.0)) * div
        import dataclasses
        gn = fb.g.norm(dim=-1).clamp_min(0.05)
        fbn = dataclasses.replace(fb, f=fb.f / gn, f_move=None)                 # distance estimate f/|grad f|: steepness cannot inflate it
        divn, _ = L.nearest_pool(fbn, fb.f.new_zeros((0, fb.f.shape[1] - 1)), torch.arange(1, fb.f.shape[1], device=fb.f.device), cfg)
        terms = {"div": div, "eik": me * L.eikonal(fb), "cons": mc * L.consistency(fb),
                 "divn": float(getattr(cfg, "variety_weight", 1.0)) * divn}
        g = {k: flat(torch.autograd.grad(v_, params, retain_graph=True, allow_unused=True), params) for k, v_ in terms.items()}
        del fb
        for space, s in (("raw", 1.0), ("adam", pre)):
            d, e, c = g["div"] * s, g["eik"] * s, g["cons"] * s
            f = e + c
            dot = float(f @ d)
            removed = max(0.0, -dot) / float(d.norm()) / float(f.norm())           # share of |field grad| PCGrad removes
            loss_drop = -dot / float(d.norm() ** 2)                                  # field step's effect on diversity, in diversity-step units
            dn = g["divn"] * s
            rows.append(dict(space=space, n_cos_eik=cos(e, dn), n_cos_field=cos(f, dn), n_vs_div=cos(dn, d),
                             n_undo=-float(f @ dn) / float(dn.norm() ** 2), cos_eik=cos(e, d), cos_cons=cos(c, d), cos_field=cos(f, d), removed=removed,
                             size_ratio=float(f.norm() / d.norm()), undo=loss_drop))
    print(f"\n{label}: multipliers eikonal {me:.1f}, consistency {mc:.1f}")
    for space in ("raw", "adam"):
        r = [x for x in rows if x["space"] == space]
        m = lambda k: sum(x[k] for x in r) / len(r)
        print(f"  {space:4s}: cos(eikonal, diversity) {m('cos_eik'):+.3f}  cos(consistency, diversity) {m('cos_cons'):+.3f}  "
              f"cos(field terms, diversity) {m('cos_field'):+.3f}  |field| / |diversity| {m('size_ratio'):.2f}  "
              f"PCGrad would remove {100 * m('removed'):.1f}% of the field gradient; field step undoes {100 * m('undo'):.1f}% of a diversity step")
        print(f"        f/|grad| diversity: cos with plain diversity {m('n_vs_div'):+.3f}; cos(eikonal, it) {m('n_cos_eik'):+.3f}  "
              f"cos(field terms, it) {m('n_cos_field'):+.3f}; field step undoes {100 * m('n_undo'):.1f}% of its step")
