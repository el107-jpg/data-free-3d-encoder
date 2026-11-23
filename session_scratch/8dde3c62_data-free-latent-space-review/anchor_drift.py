"""Are the anchor codes drifting into the directions the decoder already uses?

For each checkpoint: the usage of each code-space direction (trainer's own measure), then the share of
the anchor codes' length that lies inside the k most-used directions. Perpendicular codes start at
k / 32; a rising share means the codes are moving into the used set.
usage: anchor_drift.py <k> <label> <ckpt> [<label> <ckpt> ...]
"""
import sys

import torch

from arm_planner.data_free.train import Trainer

k = int(sys.argv[1])
for label, ckpt in zip(sys.argv[2::2], sys.argv[3::2]):
    trainer = Trainer.from_checkpoint(ckpt, device="cuda", tensorboard=False, overrides=["weak_probe_codes=32"])
    usage = trainer.measure_usage()
    lam, vec = torch.linalg.eigh(usage)
    top = vec[:, -k:].float()
    codes = torch.nn.functional.normalize(trainer.anchor_codes.detach().cpu(), dim=-1)
    share = (codes @ top).square().sum(dim=1)
    cosine = codes @ codes.T
    off = cosine[~torch.eye(codes.shape[0], dtype=torch.bool)]
    rank = torch.linalg.svdvals(codes)
    part = rank.square() / rank.square().sum()
    print(f"{label} (update {trainer.step_count}): share of anchor codes inside the top {k} used directions "
          f"mean {float(share.mean()):.3f} (min {float(share.min()):.3f}, max {float(share.max()):.3f}; start {k / 32:.3f}); "
          f"code cosines |mean| {float(off.abs().mean()):.3f} max {float(off.max()):.3f}; "
          f"effective dimensions the codes span {float(torch.exp(-(part * torch.log(part + 1e-30)).sum())):.1f} of 32")
    del trainer
    torch.cuda.empty_cache()
