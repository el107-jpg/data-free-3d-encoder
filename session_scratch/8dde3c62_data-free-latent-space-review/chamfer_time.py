"""Time and typical values of the local chamfer term on a real model (2 clusters of 8 codes)."""
import sys
import time

import torch

from arm_planner.data_free import losses as L
from arm_planner.data_free.field import evaluate_field
from arm_planner.data_free.train import Trainer

trainer = Trainer.from_checkpoint(sys.argv[1], device="cuda", tensorboard=False, overrides=["clusters=2"])
cfg, model = trainer.cfg, trainer.model
params = [p for p in model.parameters() if p.requires_grad]
times, back, vals = [], [], []
for i in range(6):
    batch = trainer.sample_batch()
    fb = evaluate_field(model, batch.codes, batch.cluster, batch.x, cfg)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    loss, info = L.local_chamfer(fb, cfg, generator=trainer.generator)
    torch.cuda.synchronize(); t1 = time.perf_counter()
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    torch.cuda.synchronize(); t2 = time.perf_counter()
    d, _ = L.diversity(fb, cfg, None)
    gd = torch.autograd.grad(d, params, allow_unused=True)
    norm = lambda gs: float(torch.sqrt(sum((g ** 2).sum() for g in gs if g is not None)))
    if i > 0:
        times.append((t1 - t0) * 1e3); back.append((t2 - t1) * 1e3)
    vals.append((float(info["local_distance"]), float(info["local_floor"]), float(info["local_above"]), norm(grads), norm(gd)))
    del fb, loss, grads
    torch.cuda.empty_cache()
m = lambda k: sum(v[k] for v in vals) / len(vals)
print(f"16 codes: forward {sum(times) / len(times):.1f} ms, its backward {sum(back) / len(back):.1f} ms (x2 for a 32-code sub-batch)")
print(f"mean surface gap {m(0):.4f}, noise floor {m(1):.5f}, share of points above the floor {m(2):.2f}")
print(f"gradient size at weight 1: {m(3):.2f}; field diversity gradient: {m(4):.2f}")
