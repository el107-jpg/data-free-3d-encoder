"""How much does SD2's surface matching change the diversity gradient? (SD2 code, its checkpoint)

Same 128 random codes and the same 2,048 shell probes (same seed); the diversity term's gradient on every model
weight with surface_match_weight = 1 (as trained) and = 0 (field distance only). Reports sizes and the angle.
"""
import copy, sys
import torch
from arm_planner.data_free.train import Trainer
from arm_planner.data_free.surface_diversity import SurfaceDiversity

path = sys.argv[1]
t = Trainer.from_checkpoint(path, device="cuda", tensorboard=False)
model, cfg = t.model, t.cfg
params = [p for p in model.parameters()]
names = [n for n, _ in model.named_parameters()]
codes = torch.nn.functional.normalize(torch.randn(128, cfg.latent_dim, generator=torch.Generator().manual_seed(11)), dim=-1).cuda()

def grad(weight):
    c = copy.copy(cfg); c.surface_match_weight = weight
    model.zero_grad(set_to_none=True)
    sd = SurfaceDiversity(model, codes, c, torch.Generator().manual_seed(5), t.step_count, defer_score=False)
    (c.variety_weight * 0 + 1)
    sd.backward()
    g = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in params]).clone()
    groups = {}
    for n, p in zip(names, params):
        key = "side network + FiLM" if n.startswith(("mapping", "film")) else ("final layer" if n.startswith("out") else "main network hidden")
        v = (p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
        groups.setdefault(key, []).append(v.clone())
    return g, {k: torch.cat(v) for k, v in groups.items()}, float(sd.value), sd.stats

g1, gr1, v1, s1 = grad(1.0)
g0, gr0, v0, s0 = grad(0.0)
cos = lambda a, b: float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-30))
diff = g1 - g0
print(f"update {t.step_count}: loss value with match {v1:.3f}, without {v0:.3f}; match distance {s1.get('surface/match_distance', float('nan')):.4f}, field distance {s1.get('surface/field_distance', float('nan')):.4f}")
print(f"gradient size: with match {g1.norm():.4f}, without {g0.norm():.4f}; ratio {g1.norm() / g0.norm():.3f}")
print(f"cosine(with, without) = {cos(g1, g0):.3f}")
print(f"match's own contribution (with - without): size {diff.norm():.4f} = {diff.norm() / g1.norm():.2f} of the full gradient; cosine with the field-only gradient {cos(diff, g0):.3f}")
for k in gr1:
    a, b = gr1[k], gr0[k]
    print(f"  {k:22s} cos {cos(a, b):.3f}  size ratio {a.norm() / b.norm().clamp_min(1e-30):.3f}  match part / full {(a - b).norm() / a.norm().clamp_min(1e-30):.2f}")
