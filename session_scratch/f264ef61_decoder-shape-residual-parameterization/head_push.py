"""Which loss terms push the learned base head, and which way? One real J4 update (CPU), warm start, ramp 1.

Prints, per term (weight included): gradient on the head bias (the push shared by all codes) for
u1, u2, u3 (extents) and u_r (radius); the descent step moves u AGAINST these signs. Also the size of
the gradient on the head weights (the part that can differ between codes).
"""
import json, sys, torch
from arm_planner.data_free import losses as L
from arm_planner.data_free import train as T
from arm_planner.data_free.config import DataFreeConfig

J4 = "/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/J4_d64_nocentre_procedural96_even_code_leak05_beta100_fourier3_5k"
cfg = T.apply_overrides(DataFreeConfig.from_dict(json.load(open(J4 + "/config.json"))),
                        ["base_shape=rounded_box", f"init_model_from={J4}/checkpoint_last.pt", "base_ramp_steps=0"] + sys.argv[1:])
trainer = T.Trainer(cfg, device="cpu", tensorboard=False)
head = trainer.model.base_head
store = {}

def take(name, term, weight):
    if not torch.is_tensor(term) or not term.requires_grad:
        return
    gb, gw = torch.autograd.grad(weight * term, [head.bias, head.weight], retain_graph=True, allow_unused=True)
    if gb is None:
        return
    b, w = store.get(name, (0, 0))
    store[name] = (b + gb / cfg.accumulate, w + gw / cfg.accumulate)

combine = trainer.multipliers.combine
def patched(objective, constraints):
    take("variety", objective, 1.0)
    for k, v in constraints.items():
        take(k, v, trainer.multipliers.multiplier(k))
    return combine(objective, constraints)
trainer.multipliers.combine = patched
am, ci = L.anchor_match, L.code_isotropy
def pa(*a, **k):
    v, i = am(*a, **k); take("anchor", v, float(cfg.anchor_weight)); return v, i
def pi(*a, **k):
    v, r = ci(*a, **k); take("iso", v, float(cfg.code_isotropy_weight)); return v, r
L.anchor_match, L.code_isotropy = pa, pi
record = trainer.step()
total_b = sum(b for b, _ in store.values())
print(f"field error {record.get('e'):.3f}  graph {record.get('graph_mode')}")
print(f"{'term':16s} {'u1':>9s} {'u2':>9s} {'u3':>9s} {'u_r':>9s}   |weight grad|")
for name, (b, w) in sorted(store.items(), key=lambda kv: -float(kv[1][0].abs().sum())):
    if float(b.abs().sum()) < 1e-6 and float(w.norm()) < 1e-6:
        continue
    print(f"{name:16s} " + " ".join(f"{float(v):9.4f}" for v in b) + f"   {float(w.norm()):9.4f}")
print(f"{'TOTAL':16s} " + " ".join(f"{float(v):9.4f}" for v in total_b) + f"   {float(sum(w for _, w in store.values()).norm()):9.4f}")
