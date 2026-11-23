"""Gradient size and batch-to-batch consistency of the statistic repulsion vs the field diversity."""
import sys

import torch

from arm_planner.data_free import losses as L
from arm_planner.data_free.field import evaluate_field
from arm_planner.data_free.train import Trainer

trainer = Trainer.from_checkpoint(sys.argv[1], device="cuda", tensorboard=False)
cfg, model = trainer.cfg, trainer.model
model.train()
params = [p for p in model.parameters() if p.requires_grad]


def flat_grad(value):
    grads = torch.autograd.grad(value, params, retain_graph=True, allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(grads, params)])


div, rep, spr, infos = [], [], [], []
for _ in range(6):
    batch = trainer.sample_batch()
    fb = evaluate_field(model, batch.codes, batch.cluster, batch.x, cfg)
    d, _ = L.diversity(fb, cfg, None)
    r, info = L.statistic_repulsion(fb, cfg)
    s, _ = L.spread(fb, cfg)
    div.append(flat_grad(d)); rep.append(flat_grad(r)); spr.append(flat_grad(s))
    infos.append({k: float(v) for k, v in info.items()})
    del fb
    torch.cuda.empty_cache()


def consistency(gs):
    cs = [float((gs[i] @ gs[j]) / (gs[i].norm() * gs[j].norm())) for i in range(len(gs)) for j in range(i + 1, len(gs))]
    return sum(cs) / len(cs)


norm = lambda gs: sum(float(g.norm()) for g in gs) / len(gs)
print(f"field diversity gradient: size {norm(div):.3f}, batch-to-batch cosine {consistency(div):.3f}")
print(f"spread term gradient:     size {norm(spr):.3f}, batch-to-batch cosine {consistency(spr):.3f}")
print(f"statistic repulsion:      size {norm(rep):.3f}, batch-to-batch cosine {consistency(rep):.3f}")
mean_g = lambda gs: torch.stack(gs).mean(0)
c = float((mean_g(rep) @ mean_g(div)) / (mean_g(rep).norm() * mean_g(div).norm()))
print(f"cosine between mean repulsion gradient and mean field-diversity gradient: {c:.3f}")
print("group spreads:", {k: round(sum(i[k] for i in infos) / len(infos), 4) for k in infos[0]})
