"""Which anchor codes have an exploded field, and where?   usage: anchor_blowup.py <checkpoint> [...]"""
import sys

import torch

from arm_planner.data_free.train import Trainer

for path in sys.argv[1:]:
    trainer = Trainer.from_checkpoint(path, device="cuda", tensorboard=False)
    model = trainer.model.eval()
    codes = torch.nn.functional.normalize(trainer.anchor_codes.detach(), dim=-1)
    gen = torch.Generator().manual_seed(3)
    x = (torch.rand(1, 40000, 3, generator=gen) * 2 - 1).cuda()
    rho = x.norm(dim=-1)
    worst = []
    with torch.no_grad():
        for k in range(codes.shape[0]):
            f = model(codes[k:k + 1], x)[0]
            worst.append((float(f.abs().max()), k, float(f.min()), float(f.max()), float(rho[0, f.argmin()])))
        big = max(float(p.abs().max()) for p in model.parameters())
    worst.sort(reverse=True)
    print(f"{path.split('/')[-1]} (update {trainer.step_count}): largest weight {big:.2f}; anchors with the largest |field| in the box:")
    for v, k, lo, hi, r in worst[:4]:
        print(f"   anchor {k:2d}: |f| max {v:.3f} (min {lo:.3f} at radius {r:.2f}, max {hi:.3f})")
    del trainer, model
    torch.cuda.empty_cache()
