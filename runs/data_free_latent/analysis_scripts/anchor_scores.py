"""Overlap of every anchor object with what the decoder makes at its code, and how it relates to the object's size.

usage: anchor_scores.py <checkpoint> [<checkpoint> ...]
"""
import sys

import torch

from arm_planner.data_free import losses as L
from arm_planner.data_free.train import Trainer

gen = torch.Generator().manual_seed(4)
x = (torch.rand(60000, 3, generator=gen) * 2 - 1).cuda()
for path in sys.argv[1:]:
    trainer = Trainer.from_checkpoint(path, device="cuda", tensorboard=False)
    model = trainer.model.eval()
    codes = torch.nn.functional.normalize(trainer.anchor_codes.detach(), dim=-1)
    grids = trainer.anchor_grid.float()
    ious, volumes = [], []
    with torch.no_grad():
        for k in range(codes.shape[0]):
            ours = model(codes[k:k + 1], x[None])[0] > 0
            theirs = L.sample_grid(grids[k:k + 1], x[None])[0] > 0
            ious.append(float((ours & theirs).sum() / (ours | theirs).sum().clamp_min(1)))
            volumes.append(float(theirs.float().mean()) * 8.0)                 # volume of the object (the cube has volume 8)
    iou, vol = torch.tensor(ious), torch.tensor(volumes)
    order = torch.argsort(iou)
    print(f"\n{path.split('/')[-2]} (update {trainer.step_count}): {len(ious)} anchors; overlap mean {float(iou.mean()):.3f}, median {float(iou.median()):.3f}")
    print(f"  at or above 0.95: {int((iou >= 0.95).sum())}; 0.90 to 0.95: {int(((iou >= 0.9) & (iou < 0.95)).sum())}; "
          f"0.80 to 0.90: {int(((iou >= 0.8) & (iou < 0.9)).sum())}; 0.50 to 0.80: {int(((iou >= 0.5) & (iou < 0.8)).sum())}; below 0.50: {int((iou < 0.5).sum())}")
    print("  the eight worst (overlap, object volume):", ", ".join(f"{float(iou[i]):.2f} / {float(vol[i]):.3f}" for i in order[:8]))
    small = vol < vol.median()
    print(f"  smaller half of the objects (volume under {float(vol.median()):.3f}): mean overlap {float(iou[small].mean()):.3f}; larger half: {float(iou[~small].mean()):.3f}")
