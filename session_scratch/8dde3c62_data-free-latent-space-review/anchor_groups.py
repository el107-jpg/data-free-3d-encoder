"""Anchor overlap by group for the 96-anchor set: thick (0-63), thin rods (64-79), thin sheets (80-95).

usage: anchor_groups.py <checkpoint> [<checkpoint> ...]
Also counts, for each decoded anchor, the number of separate pieces on a 96-cell grid (a rod turned into a
string of beads shows as several pieces).
"""
import sys

import numpy as np
import scipy.ndimage as ndi
import torch

from arm_planner.data_free import losses as L
from arm_planner.data_free.train import Trainer

gen = torch.Generator().manual_seed(4)
x = (torch.rand(60000, 3, generator=gen) * 2 - 1).cuda()
axis = torch.linspace(-1, 1, 96, device="cuda")
grid_pts = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(1, -1, 3)
for path in sys.argv[1:]:
    trainer = Trainer.from_checkpoint(path, device="cuda", tensorboard=False)
    model = trainer.model.eval()
    codes = torch.nn.functional.normalize(trainer.anchor_codes.detach(), dim=-1)
    grids = trainer.anchor_grid.float()
    ious, pieces = [], []
    with torch.no_grad():
        for k in range(codes.shape[0]):
            ours = model(codes[k:k + 1], x[None])[0] > 0
            theirs = L.sample_grid(grids[k:k + 1], x[None])[0] > 0
            ious.append(float((ours & theirs).sum() / (ours | theirs).sum().clamp_min(1)))
            solid = torch.cat([model(codes[k:k + 1], grid_pts[:, s:s + 250000])[0] for s in range(0, grid_pts.shape[1], 250000)])
            pieces.append(int(ndi.label((solid > 0).reshape(96, 96, 96).cpu().numpy())[1]))
    iou, pieces = np.array(ious), np.array(pieces)
    print(f"\n{path.split('/')[-2]} (update {trainer.step_count})")
    groups = [("all", slice(0, len(iou)))]
    if len(iou) == 96:
        groups += [("thick, 0-63", slice(0, 64)), ("thin rods, 64-79", slice(64, 80)), ("thin sheets, 80-95", slice(80, 96))]
    for label, s in groups:
        v, p = iou[s], pieces[s]
        print(f"  {label:20s} overlap mean {v.mean():.3f}, median {np.median(v):.3f}, worst {v.min():.2f}; at or above 0.80: {int((v >= 0.8).sum())} of {len(v)}; "
              f"below 0.50: {int((v < 0.5).sum())}; decoded as more than one piece: {int((p > 1).sum())}; empty: {int((p == 0).sum())}")
