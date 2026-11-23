"""How much of the loss and of the gradient comes from the phase 2 shape statistics?

The statistics enter inside the pair distance (distance = field distance + weight * statistic gap),
which then goes through the floor and ordering terms. So:
  - share of the DISTANCE: weight * gap against the total, per pair;
  - share of the LOSS: diversity loss with the statistics on, against the same batch with them off;
  - share of the GRADIENT: gradient of the diversity loss on the decoder parameters, split into the
    path through the statistics and the path through the field distance (statistics detached).
Spread term and the weighted constraints are reported for scale.
"""
import json
import sys

import torch

from arm_planner.data_free import losses as L
from arm_planner.data_free.field import evaluate_field
from arm_planner.data_free.train import Trainer

ckpt, batches = sys.argv[1], int(sys.argv[2])
trainer = Trainer.from_checkpoint(ckpt, device="cuda", tensorboard=False, overrides=["clusters=2"])
cfg, model = trainer.cfg, trainer.model
model.train()
params = [p for p in model.parameters() if p.requires_grad]
step_on = int(cfg.phase2_start_step) + 1
real_descriptors = L.descriptors


def flat_grad(value):
    grads = torch.autograd.grad(value, params, retain_graph=True, allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(grads, params)])


rows = []
for _ in range(batches):
    batch = trainer.sample_batch()
    fb = evaluate_field(model, batch.codes, batch.cluster, batch.x, cfg)

    L.descriptors = real_descriptors
    div_on, info_on = L.diversity(fb, cfg, step_on)
    div_off, info_off = L.diversity(fb, cfg, None)
    g_full = flat_grad(div_on)
    L.descriptors = lambda f_, c_: real_descriptors(f_, c_).detach()        # statistics path cut
    div_cut, _ = L.diversity(fb, cfg, step_on)
    g_field = flat_grad(div_cut)
    L.descriptors = real_descriptors
    g_stats = g_full - g_field

    spread, _ = L.spread(fb, cfg)
    g_spread = float(cfg.spread_weight) * flat_grad(spread)
    cons = {"eikonal": L.eikonal(fb), "consistency": L.consistency(fb), "barrier": L.barrier(fb, cfg),
            "origin": L.origin_anchor(fb, cfg), "volume": L.min_volume(fb, cfg)}
    cons.update(L.pose_terms(fb, cfg))
    weighted = sum(trainer.multipliers.multiplier(k) * v for k, v in cons.items())
    g_cons = flat_grad(weighted)
    g_total = g_full + g_spread + g_cons

    # the statistic gap itself, per pair
    with torch.no_grad():
        stats = real_descriptors(fb, cfg)
        scales = torch.as_tensor(L.DESCRIPTOR_SCALES, device=stats.device)
        same = fb.cluster[:, None] == fb.cluster[None, :]
        a, b = torch.nonzero(same & torch.triu(torch.ones_like(same), 1), as_tuple=True)
        per_stat = ((stats[a] - stats[b]).abs() / scales).mean(0)
        gap = ((stats[a] - stats[b]).abs() / scales).mean(1)
        added = float(cfg.descriptor_weight) * gap

    cos = lambda u, v: float((u @ v) / (u.norm() * v.norm() + 1e-30))
    rows.append({
        "distance_total": float(info_on["mean_shape_distance"]),
        "distance_from_stats": float(added.mean()),
        "div_loss_on": float(div_on), "div_loss_off": float(div_off),
        "violations_on": float(info_on["order_violations"]), "violations_off": float(info_off["order_violations"]),
        "spread_weighted": float(cfg.spread_weight) * float(spread), "constraints_weighted": float(weighted),
        "g_div": float(g_full.norm()), "g_div_field_path": float(g_field.norm()),
        "g_div_stats_path": float(g_stats.norm()), "g_spread": float(g_spread.norm()),
        "g_constraints": float(g_cons.norm()), "g_total": float(g_total.norm()),
        "cos_stats_vs_field": cos(g_stats, g_field), "cos_stats_vs_total": cos(g_stats, g_total),
        "stats_projection_on_total": float((g_stats @ g_total) / (g_total @ g_total + 1e-30)),
        "per_stat_gap": per_stat.tolist(),
    })
    del fb
    torch.cuda.empty_cache()

mean = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0] if k != "per_stat_gap"}
per_stat = [sum(r["per_stat_gap"][i] for r in rows) / len(rows) for i in range(len(L.DESCRIPTOR_NAMES))]
print("step", trainer.step_count, "| descriptor_weight", cfg.descriptor_weight, "| batches", batches)
for k, v in mean.items():
    print(f"  {k:28s} {v:.5g}")
print("  statistic gap per pair, in units of its scale:")
for n, v in zip(L.DESCRIPTOR_NAMES, per_stat):
    print(f"    {n:16s} {v:.3f}")
json.dump({"step": trainer.step_count, "mean": mean, "per_stat": dict(zip(L.DESCRIPTOR_NAMES, per_stat))},
          open("/home/eric-luo/.claude/jobs/8dde3c62/tmp/phase2_share.json", "w"), indent=1)
