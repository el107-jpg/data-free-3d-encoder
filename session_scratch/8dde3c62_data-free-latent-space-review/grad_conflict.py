"""Do the loss terms pull the decoder's weights in opposing directions?

On a saved model: several small batches (random codes plus a few anchor codes), the gradient of each
term with respect to all decoder weights, then the cosine between terms and how much of the variety
push each constraint cancels at its current weight.
usage: grad_conflict.py <checkpoint> [batches]
"""
import sys

import torch

from arm_planner.data_free import losses as L
from arm_planner.data_free.field import evaluate_field, sample_points
from arm_planner.data_free.latent import sample_sphere
from arm_planner.data_free.train import Trainer

ckpt = sys.argv[1]
batches = int(sys.argv[2]) if len(sys.argv) > 2 else 6
trainer = Trainer.from_checkpoint(ckpt, device="cuda", tensorboard=False, overrides=["points=1024"])
cfg, model = trainer.cfg, trainer.model
params = [p for p in model.parameters()]
gen = torch.Generator().manual_seed(123)
names = ["variety", "anchor pull", "consistency", "eikonal", "pose", "barrier"]
dots = torch.zeros(len(names), len(names), dtype=torch.float64)
norms = torch.zeros(len(names), dtype=torch.float64)
cos_sum = torch.zeros(len(names), len(names), dtype=torch.float64)
by_group = {}


def flat(value):
    got = torch.autograd.grad(value, params, retain_graph=True, allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(got, params)]).double()


for _ in range(batches):
    free = sample_sphere(10, cfg.latent_dim, gen, "cuda")
    pick = torch.randperm(trainer.anchor_codes.shape[0], generator=gen)[:4].cuda()
    codes = torch.cat([free, torch.nn.functional.normalize(trainer.anchor_codes.detach()[pick], dim=-1)])
    x = sample_points(cfg, 1, gen, "cuda")
    fb = evaluate_field(model, codes, torch.zeros(14, dtype=torch.long, device="cuda"), x, cfg)
    main = type(fb)(codes=fb.codes[:10], cluster=fb.cluster[:10], x=fb.x[:10], f=fb.f[:10], g=fb.g[:10],
                    xp=fb.xp[:10], fp=fb.fp[:10])
    index = torch.arange(1, x.shape[1], device="cuda")
    variety, _ = L.nearest_pool(main, fb.f.new_zeros((0, index.numel())), index, cfg)
    anchor, _ = L.anchor_match(fb, torch.arange(10, 14, device="cuda"), trainer.anchor_grid[pick].float(), cfg)
    pose = sum(L.pose_terms(fb, cfg).values())
    terms = [variety, anchor, L.consistency(main), L.eikonal(main), pose, L.barrier(fb, cfg)]
    grads = torch.stack([flat(t) for t in terms])
    gram = grads @ grads.T
    size = gram.diagonal().clamp_min(1e-30).sqrt()
    dots += gram.cpu() / batches
    norms += size.cpu() / batches
    cos_sum += (gram / (size[:, None] * size[None, :])).cpu() / batches
    # the same by part of the network: where is the conflict?
    start = 0
    for (name, p) in model.named_parameters():
        group = name.split(".")[0]
        part = grads[:, start:start + p.numel()]
        start += p.numel()
        acc = by_group.setdefault(group, torch.zeros(3, dtype=torch.float64))
        acc += torch.stack([(part[0] * part[2]).sum(), part[0].square().sum(), part[2].square().sum()]).cpu()
    del fb, grads

print(f"{ckpt.split('/')[-2]} update {trainer.step_count}; {batches} batches of 10 random + 4 anchor codes")
print("gradient size of each term (unweighted):", {n: round(float(v), 4) for n, v in zip(names, norms)})
mult = {"consistency": trainer.multipliers.multiplier("consistency"), "eikonal": trainer.multipliers.multiplier("eikonal"),
        "barrier": trainer.multipliers.multiplier("barrier")}
print("current multipliers:", {k: round(v, 1) for k, v in mult.items()}, "| anchor weight", cfg.anchor_weight)
print("\ncosine between term gradients (mean over batches; negative = they pull the weights apart):")
print(" " * 14 + "".join(f"{n[:11]:>12s}" for n in names))
for i, n in enumerate(names):
    print(f"{n:14s}" + "".join(f"{float(cos_sum[i, j]):12.3f}" for j in range(len(names))))
print("\nvariety against consistency, by part of the network (cosine; share of each gradient's energy in that part):")
tot_v = sum(float(v[1]) for v in by_group.values())
tot_c = sum(float(v[2]) for v in by_group.values())
for group, v in by_group.items():
    print(f"  {group:8s} cosine {float(v[0] / (v[1] * v[2]).sqrt()):+.3f}   variety energy {float(v[1]) / tot_v:.3f}   "
          f"consistency energy {float(v[2]) / tot_c:.3f}")
print("\nalong the variety direction, each other term at its current weight (negative = cancels the variety push, positive = adds to it):")
weights = {"anchor pull": float(cfg.anchor_weight), "consistency": mult["consistency"], "eikonal": mult["eikonal"],
           "pose": 50.0, "barrier": mult["barrier"]}
for j, n in enumerate(names[1:], start=1):
    share = weights[n] * float(dots[0, j]) / float(dots[0, 0])
    print(f"  {n:12s} weight {weights[n]:7.1f}: {share:+.2f} of the variety gradient's own size along its direction")
