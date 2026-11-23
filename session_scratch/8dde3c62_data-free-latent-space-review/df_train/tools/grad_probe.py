"""Gradient of each loss term at a checkpoint: norm, and agreement across two point samples."""
import sys
import torch
from arm_planner.data_free import losses as L
from arm_planner.data_free.field import evaluate_field, sample_points
from arm_planner.data_free.train import Trainer

tr = Trainer.from_checkpoint(sys.argv[1], device="cuda", tensorboard=False)
cfg, model = tr.cfg, tr.model
params = [p for p in model.parameters()]
print("multipliers", {k: round(v, 3) for k, v in tr.multipliers.multipliers().items() if v != 0.01})


def grads(batch, x):
    fb = evaluate_field(model, batch.codes, batch.cluster, x, cfg)
    div, info = L.diversity(fb, cfg, tr.step_count)
    terms = {"diversity": div, "eikonal": L.eikonal(fb), "consistency": L.consistency(fb)}
    terms.update(L.pose_terms(fb, cfg))
    out = {}
    for k, v in terms.items():
        g = torch.autograd.grad(v, params, retain_graph=True, allow_unused=True)
        out[k] = torch.cat([(t if t is not None else torch.zeros_like(p)).reshape(-1) for t, p in zip(g, params)])
    return out, {k: float(v) for k, v in terms.items()}, float(info["mean_ratio"])


keys = ["diversity", "eikonal", "consistency", "pose_centroid", "pose_scale", "pose_axes"]
norms = {k: [] for k in keys}; cos = {k: [] for k in keys}; mean_g = {k: 0 for k in keys}
nb = 6
for i in range(nb):
    batch = tr.sample_batch()
    ga, va, ra = grads(batch, batch.x)
    gb, vb, rb = grads(batch, sample_points(cfg, cfg.clusters, tr.generator, tr.device))
    for k in keys:
        norms[k].append(float(ga[k].norm()))
        cos[k].append(float(torch.nn.functional.cosine_similarity(ga[k], gb[k], dim=0)))
        mean_g[k] = mean_g[k] + (ga[k] + gb[k]) / (2 * nb)
print("term: |grad| (unweighted, mean over batches) | cosine between two point samples of the same codes | |mean grad over 12 samples|")
for k in keys:
    n = sum(norms[k]) / nb
    print(f"  {k:14s} {n:10.3e}   cos {sum(cos[k]) / nb:6.3f}   |mean| {float(mean_g[k].norm()):10.3e}")
