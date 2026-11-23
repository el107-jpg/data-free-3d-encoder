"""Which loss term makes the shapes smaller?

On a saved model, for several small batches (random codes plus a few anchor codes): the gradient of every
term (times its current weight) with respect to the decoder's weights, and the gradient of two probes of
the random codes' shapes: their mean soft volume and their mean size^2 under the run's pose rule. A plain
gradient step on a term changes a probe by -(term gradient . probe gradient); reported per term, as a
fraction of the probe's value per unit step, and scaled so the largest is 1.
usage: fork_shrink_probe.py <checkpoint> [batches]
"""
import sys

import torch

from arm_planner.data_free import losses as L
from arm_planner.data_free.field import evaluate_field, sample_points
from arm_planner.data_free.latent import sample_sphere
from arm_planner.data_free.train import Trainer, _first_rows

ckpt = sys.argv[1]
batches = int(sys.argv[2]) if len(sys.argv) > 2 else 8
trainer = Trainer.from_checkpoint(ckpt, device="cuda", tensorboard=False, overrides=["points=1024"])
cfg, model = trainer.cfg, trainer.model
params = [p for p in model.parameters()]
gen = torch.Generator().manual_seed(123)
power = float(cfg.pose_power)
M = trainer.multipliers.multiplier
names = ["variety", "anchor pull", "consistency", "eikonal", "pose centre", "pose size", "pose axes", "barrier"]
weights = [float(getattr(cfg, "variety_weight", 1.0)), float(cfg.anchor_weight), M("consistency"), M("eikonal"),
           M("pose_centroid"), M("pose_scale"), M("pose_axes"), M("barrier")]
effect = torch.zeros(len(names), 2, dtype=torch.float64)
value = torch.zeros(2, dtype=torch.float64)
norm = torch.zeros(len(names), dtype=torch.float64)


def flat(v):
    got = torch.autograd.grad(v, params, retain_graph=True, allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(got, params)]).double()


n_free, n_anchor = 10, 4
for _ in range(batches):
    free = sample_sphere(n_free, cfg.latent_dim, gen, "cuda")
    pick = torch.randperm(trainer.anchor_codes.shape[0], generator=gen)[:n_anchor].cuda()
    codes = torch.cat([free, torch.nn.functional.normalize(trainer.anchor_codes.detach()[pick], dim=-1)])
    x = sample_points(cfg, 1, gen, "cuda")
    fb = evaluate_field(model, codes, torch.zeros(n_free + n_anchor, dtype=torch.long, device="cuda"), x, cfg)
    main = _first_rows(fb, n_free)
    index = torch.arange(1, x.shape[1], device="cuda")
    variety, _ = L.nearest_pool(main, fb.f.new_zeros((0, index.numel())), index, cfg)
    anchor, _ = L.anchor_match(fb, torch.arange(n_free, n_free + n_anchor, device="cuda"), trainer.anchor_grid[pick].float(), cfg)
    pose = L.pose_terms(fb, cfg)
    terms = [variety, anchor, L.consistency(main), L.eikonal(main), pose["pose_centroid"], pose["pose_scale"],
             pose["pose_axes"], L.barrier(fb, cfg)]
    w = torch.sigmoid(main.f / float(cfg.soft_tau))
    volume = w[:, 1:].mean()
    _, moment = L.extent_pose(w[:, 1:], main.x[:, 1:].detach(), power)
    size_sq = moment.clamp_min(1e-12).pow(2.0 / power).mean()
    probes = torch.stack([flat(volume), flat(size_sq)])
    value += torch.tensor([float(volume), float(size_sq)], dtype=torch.float64) / batches
    for i, (t, wt) in enumerate(zip(terms, weights)):
        g = wt * flat(t)
        norm[i] += float(g.norm()) / batches
        effect[i] += -(probes @ g).cpu() / batches
    del fb

target = L.pose_size_target(cfg) ** 2
print(f"{ckpt.split('/')[-2]} update {trainer.step_count}, pose_power {power:g}; {batches} batches of {n_free} random + {n_anchor} anchor codes")
print(f"random codes now: mean soft volume share {float(value[0]):.4f}; mean size^2 {float(value[1]):.4f} (target {target:.4f})")
rel = effect / value * float(cfg.lr) * 100.0        # percent change per plain gradient step at the run's learning rate
scale = torch.ones(2, dtype=torch.float64)
print(f"{'term':13s} {'weight':>8s} {'gradient size':>14s} {'effect on volume':>18s} {'effect on size^2':>18s}   (negative = a step on this term shrinks the shapes; largest = 1)")
for i, n in enumerate(names):
    print(f"{n:13s} {weights[i]:8.1f} {float(norm[i]):14.3f} {float(rel[i, 0] / scale[0]):+18.2f} {float(rel[i, 1] / scale[1]):+18.2f}")
total = rel.sum(0)
print(f"{'all together':13s} {'':8s} {'':14s} {float(total[0] / scale[0]):+18.2f} {float(total[1] / scale[1]):+18.2f}")
