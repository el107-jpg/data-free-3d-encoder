"""Size of the crease term and of its gradient on a saved model, next to the logged gradient of the whole loss.

usage: crease_calibrate.py <checkpoint> [<checkpoint> ...]
"""
import json
import os
import sys

import numpy as np
import torch

from arm_planner.data_free import losses as L
from arm_planner.data_free.field import evaluate_field, sample_points
from arm_planner.data_free.latent import sample_sphere
from arm_planner.data_free.train import Trainer

for path in sys.argv[1:]:
    trainer = Trainer.from_checkpoint(path, device="cuda", tensorboard=False)
    cfg, model = trainer.cfg, trainer.model.train()
    gen = torch.Generator(device="cuda").manual_seed(7)
    rows = []
    for rep in range(4):
        codes = sample_sphere(24, cfg.latent_dim, gen, "cuda")
        x = sample_points(cfg, 24, gen, "cuda")
        fb = evaluate_field(model, codes, torch.arange(24, device="cuda"), x, cfg)
        out = {}
        value, info = L.crease(model, fb, cfg, generator=gen)
        grads = torch.autograd.grad(value, list(model.parameters()), retain_graph=True, allow_unused=True)
        out["crease"] = float(value)
        out["crease_grad"] = float(torch.sqrt(sum((g ** 2).sum() for g in grads if g is not None)))
        out.update({k: float(v) for k, v in info.items()})
        smooth = L.grad_change(model, fb, cfg, generator=gen)
        grads = torch.autograd.grad(smooth, list(model.parameters()), allow_unused=True)
        out["smooth"] = float(smooth)
        out["smooth_grad"] = float(torch.sqrt(sum((g ** 2).sum() for g in grads if g is not None)))
        rows.append(out)
        del fb, grads
    mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    log = os.path.join(os.path.dirname(path), "metrics.jsonl")
    norms = []
    for line in open(log):
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if abs(rec.get("step", -10 ** 9) - trainer.step_count) <= 200 and "grad_norm" in rec:
            norms.append(rec["grad_norm"])
    print(f"{path.split('/')[-2][:30]} @ {trainer.step_count}: crease term {mean['crease']:.3f} (pairs used {mean['used']:.2f}, median bend {mean['turn']:.2f}, "
          f"flat share {mean['flat']:.2f}, sharp share {mean['sharp']:.2f}); gradient of the term per unit weight {mean['crease_grad']:.3f}; "
          f"existing smoothness term {mean['smooth']:.4f}, its gradient per unit weight {mean['smooth_grad']:.3f} (times weight {cfg.grad_change_weight}: {mean['smooth_grad'] * cfg.grad_change_weight:.5f}); "
          f"logged gradient of the whole loss near this update: median {np.median(norms) if norms else float('nan'):.3f} (clip {cfg.grad_clip})", flush=True)
