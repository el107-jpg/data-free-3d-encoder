"""Per-term push on the bend/taper outputs of the base head (B16), one real update on the CPU.

For every loss term (its weight included): the gradient on the last head layer's bias rows (push shared by all
codes) and the norm of the gradient on its weight rows (push that can differ between codes), for the shape rows
(extents + radius), the bend rows and the taper rows. Descent moves the outputs AGAINST the bias gradient.
"""
import sys, torch
from arm_planner.data_free import losses as L
from arm_planner.data_free.train import Trainer

t = Trainer.from_checkpoint(sys.argv[1], device="cpu", tensorboard=False)
last = t.model.base_head_mlp[-1]
store = {}

def take(name, value, weight=1.0):
    if not torch.is_tensor(value) or not value.requires_grad:
        return
    gw, gb = torch.autograd.grad(weight * value, [last.weight, last.bias], retain_graph=True, allow_unused=True)
    if gw is None:
        return
    a = store.setdefault(name, [0, 0])
    a[0] = a[0] + gw / t.cfg.accumulate
    a[1] = a[1] + gb / t.cfg.accumulate

combine = t.multipliers.combine
def comb(objective, constraints):
    take("variety", objective)
    for k, v in constraints.items():
        take(k, v, t.multipliers.multiplier(k))
    return combine(objective, constraints)
t.multipliers.combine = comb
am, br, ci = L.anchor_match, L.base_repel, L.code_isotropy
def pa(*a, **k):
    v, i = am(*a, **k); take("anchor", v, float(t._anchor_weight(t.step_count))); return v, i
def pr(*a, **k):
    v, i = br(*a, **k); take("base_repel", v, float(t.cfg.base_repel_weight)); return v, i
def pc(*a, **k):
    v, r = ci(*a, **k); take("iso", v, float(t.cfg.code_isotropy_weight)); return v, r
L.anchor_match, L.base_repel, L.code_isotropy = pa, pr, pc
cc, bt = t._correction_cost, t._base_terms
def pcc(fb, rows=None):
    v, i = cc(fb, rows); take("correction_cost", v, float(t.cfg.correction_cost_weight)); return v, i
def pbt(fb):
    dev, neg, i = bt(fb)
    take("base_budget", dev, t.multipliers.multiplier("base_deviation")); take("base_negative", neg, float(t.cfg.base_negative_weight))
    return dev, neg, i
t._correction_cost, t._base_terms = pcc, pbt
t.optimizer.step = lambda *a, **k: None
rec = t.step()
print(f"update {t.step_count - 1}: loss {rec['loss']:.2f}; base head lr {t.optimizer.param_groups[1]['lr']:.2e}")
rows = {"shape (b,r)": slice(0, 4), "bends": slice(4, 10), "tapers": slice(10, 13)}
print(f"{'term':16s}" + "".join(f"{n:>26s}" for n in rows))
print(f"{'':16s}" + "".join(f"{'shared  per-code':>26s}" for n in rows))
tot_w, tot_b = 0, 0
for name, (gw, gb) in sorted(store.items(), key=lambda kv: -float(kv[1][0][4:10].norm())):
    tot_w, tot_b = tot_w + gw, tot_b + gb
    if float(gw.norm()) < 1e-7:
        continue
    print(f"{name:16s}" + "".join(f"{float(gb[s].abs().mean()):13.4f}{float(gw[s].norm()):13.4f}" for s in rows.values()))
print(f"{'TOTAL':16s}" + "".join(f"{float(tot_b[s].abs().mean()):13.4f}{float(tot_w[s].norm()):13.4f}" for s in rows.values()))
with torch.no_grad():
    z = torch.nn.functional.normalize(torch.randn(512, t.cfg.latent_dim, generator=torch.Generator().manual_seed(0)), dim=-1)
    d = t.model.prepare(z).deform
print("bend |kappa| mean %.4f max %.4f; taper |t| mean %.4f; head bend-row weight norm %.4f, bias %s" % (
    float(d[:, :6].abs().mean()), float(d[:, :6].abs().max()), float(d[:, 6:9].abs().mean()),
    float(last.weight[4:10].norm()), [round(float(v), 4) for v in last.bias[4:10]]))
