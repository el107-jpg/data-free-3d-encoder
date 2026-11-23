"""Can a trained decoder reproduce objects it was not trained on, if we search for the best code?

For each target object (a distance grid, positive inside, in the trainer's pose convention) a unit
code is optimised with the decoder frozen, minimising the trainer's own anchor value loss: the mean
absolute difference between the decoder's field and the target at uniform points inside the bound
radius. Several starts per target; the start with the lowest final loss (on fresh points) is kept.
Overlap = intersection over union of the two solids on 20,000 fixed uniform points.

usage: heldout_fit.py <checkpoint> <eligible.pt> <tag> [steps] [starts]
Writes <tag>.json and <tag>.pt (best codes) next to this script.
"""
import json
import math
import os
import sys

import torch

from arm_planner.data_free import losses as L
from arm_planner.data_free.train import Trainer

ckpt, eligible_path, tag = sys.argv[1], sys.argv[2], sys.argv[3]
steps = int(sys.argv[4]) if len(sys.argv) > 4 else 500
starts = int(sys.argv[5]) if len(sys.argv) > 5 else 8
BLOCK = int(sys.argv[6]) if len(sys.argv) > 6 else 8          # targets optimised together (memory)
HERE = os.path.dirname(os.path.abspath(__file__))
dev = "cuda"
trainer = Trainer.from_checkpoint(ckpt, device=dev, tensorboard=False)
cfg, model = trainer.cfg, trainer.model.eval()
for p in model.parameters():
    p.requires_grad_(False)
dim = int(cfg.latent_dim)
data = torch.load(eligible_path, map_location="cpu", weights_only=False)
grids = data["grid"].float()
chosen = torch.tensor(data["chosen"])
gen = torch.Generator().manual_seed(2026)
bound = float(cfg.bound_radius)
test_x = (torch.rand(20000, 3, generator=gen) * 2 - 1).to(dev)
test_in = test_x.square().sum(-1) < bound ** 2
has_anchors = trainer.anchor_codes is not None
anchor_codes = torch.nn.functional.normalize(trainer.anchor_codes.detach(), dim=-1) if has_anchors else None
anchor_grid = trainer.anchor_grid.float() if has_anchors else None


def score(codes, target):
    """Value loss and overlap of codes (S x D) against one target grid, on the fixed test points."""
    with torch.no_grad():
        f = torch.cat([model(codes[s:s + 16], test_x.expand(codes[s:s + 16].shape[0], -1, -1)) for s in range(0, codes.shape[0], 16)])
        t = L.sample_grid(target[None].to(dev), test_x[None])[0]
        loss = ((f - t).abs() * test_in).sum(1) / test_in.sum()
        ours, theirs = f > 0, t > 0
        iou = (ours & theirs).sum(1) / (ours | theirs).sum(1).clamp_min(1)
    return loss, iou


def fit(targets, first_codes=None):
    """Best code per target. ``first_codes`` (T x D or None) is one informed start per target."""
    count = targets.shape[0]
    targets = targets.to(dev)
    # screen 256 random codes per target, keep the best few as starts
    pool = torch.nn.functional.normalize(torch.randn(256, dim, generator=gen), dim=-1).to(dev)
    with torch.no_grad():
        fp = torch.cat([model(pool[s:s + 32], test_x[:4000].expand(pool[s:s + 32].shape[0], -1, -1)) for s in range(0, 256, 32)])
        tp = L.sample_grid(targets, test_x[:4000].expand(count, -1, -1))
        screen = torch.stack([((fp - tp[i]).abs() * test_in[:4000]).sum(1) for i in range(count)])      # T x 256
    informed = 0 if first_codes is None else 1
    best = max(1, (starts - informed) // 2)
    rand = starts - informed - best
    init = []
    for i in range(count):
        rows = [pool[screen[i].topk(best, largest=False).indices]]
        if rand > 0:
            rows.append(torch.nn.functional.normalize(torch.randn(rand, dim, generator=gen), dim=-1).to(dev))
        if informed:
            rows.append(first_codes[i:i + 1].to(dev))
        init.append(torch.cat(rows))
    codes = torch.nn.Parameter(torch.stack(init))                                               # T x S x D
    opt = torch.optim.Adam([codes], lr=0.03)
    for step in range(steps):
        for group in opt.param_groups:
            group["lr"] = 0.002 + 0.028 * 0.5 * (1 + math.cos(math.pi * step / steps))
        x = (torch.rand(1, 2048, 3, generator=gen) * 2 - 1).to(dev)
        inside = (x.square().sum(-1) < bound ** 2).float()
        with torch.no_grad():
            t = L.sample_grid(targets, x.expand(count, -1, -1))                                  # T x N
        f = model(codes.reshape(count * starts, dim), x.expand(count * starts, -1, -1)).reshape(count, starts, -1)
        loss = (((f - t[:, None]).abs() * inside).sum(-1) / inside.sum()).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    final = torch.nn.functional.normalize(codes.detach(), dim=-1)
    out_code, out_loss, out_iou, out_start = [], [], [], []
    for i in range(count):
        loss, iou = score(final[i], targets[i])
        j = int(loss.argmin())
        out_code.append(final[i, j]); out_loss.append(float(loss[j])); out_iou.append(float(iou[j])); out_start.append(j)
    return torch.stack(out_code).cpu(), out_loss, out_iou, out_start


def run(index, informed):
    codes, losses, ious, picks = [], [], [], []
    for s in range(0, len(index), BLOCK):
        block = index[s:s + BLOCK]
        targets = grids[block]
        first = None
        if informed and has_anchors:
            # the trained code of the anchor object most like the target
            with torch.no_grad():
                t = L.sample_grid(targets.to(dev), test_x[:4000].expand(len(block), -1, -1))
                a = torch.cat([L.sample_grid(anchor_grid[k:k + 16], test_x[:4000].expand(anchor_grid[k:k + 16].shape[0], -1, -1))
                               for k in range(0, anchor_grid.shape[0], 16)])
                near = ((t[:, None] - a[None]).abs() * test_in[:4000]).sum(-1).argmin(1)
            first = anchor_codes[near]
        c, l, i, p = fit(targets, first)
        codes.append(c); losses += l; ious += i; picks += p
    return torch.cat(codes), losses, ious, picks


def stats(v):
    t = torch.tensor(v)
    return {"count": len(v), "mean": float(t.mean()), "median": float(t.median()), "worst": float(t.min()),
            "best": float(t.max()), "share_at_or_above_0.70": float((t >= 0.70).float().mean())}


held = torch.nonzero(~chosen).flatten().tolist()
anch = torch.nonzero(chosen).flatten().tolist()
result = {"checkpoint": ckpt, "update": trainer.step_count, "steps": steps, "starts": starts, "has_anchors": has_anchors}
codes_h, loss_h, iou_h, pick_h = run(held, informed=True)
result["held_out"] = {"iou": stats(iou_h), "value_loss": stats(loss_h), "per_object_iou": iou_h,
                      "file_ids": [data["file_ids"][i] for i in held],
                      "informed_start_won": (sum(1 for p in pick_h if p == starts - 1) / len(pick_h)) if has_anchors else None}
print(tag, "held-out objects:", json.dumps(result["held_out"]["iou"]), flush=True)
codes_a, loss_a, iou_a, _ = run(anch, informed=False)
result["anchors_by_search"] = {"iou": stats(iou_a), "value_loss": stats(loss_a), "per_object_iou": iou_a,
                               "file_ids": [data["file_ids"][i] for i in anch]}
print(tag, "the 64 anchor objects, found by search from random starts:", json.dumps(result["anchors_by_search"]["iou"]), flush=True)
if has_anchors:
    # at the trained codes; the anchor file is in selection order, eligible.pt is not: match by file id
    order = torch.load(str(cfg.anchor_path), map_location="cpu", weights_only=False)["file_ids"]
    ious = []
    for k, fid in enumerate(order[:anchor_codes.shape[0]]):
        _, iou = score(anchor_codes[k:k + 1], anchor_grid[k])
        ious.append(float(iou[0]))
    result["anchors_at_trained_code"] = {"iou": stats(ious)}
    print(tag, "the anchor objects at their trained codes:", json.dumps(result["anchors_at_trained_code"]["iou"]), flush=True)
json.dump(result, open(os.path.join(HERE, tag + ".json"), "w"), indent=1)
torch.save({"held_codes": codes_h, "held_index": held, "anchor_codes_found": codes_a, "anchor_index": anch},
           os.path.join(HERE, tag + ".pt"))
print("saved", os.path.join(HERE, tag + ".json"))
