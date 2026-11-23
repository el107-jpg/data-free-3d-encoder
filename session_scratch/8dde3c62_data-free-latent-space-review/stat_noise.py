"""Signal versus sampling noise for each shape statistic and for the field distance.

For 64 codes, compute the statistics from several independent draws of the training number of
points. Noise = spread of one code's value across draws; signal = spread of the mean across codes.
Also: is the statistics' gradient consistent between two independent batches?
"""
import sys

import torch

from arm_planner.data_free import losses as L
from arm_planner.data_free.field import evaluate_field, sample_points
from arm_planner.data_free.train import Trainer

trainer = Trainer.from_checkpoint(sys.argv[1], device="cuda", tensorboard=False)
cfg, model = trainer.cfg, trainer.model.eval()
gen = torch.Generator().manual_seed(11)
codes = trainer.bank.codes[:64]
draws = 6
vals = []
for d in range(draws):
    out = []
    for c0 in range(0, 64, 16):
        cc = codes[c0:c0 + 16]
        x = sample_points(cfg, 16, gen, "cuda")
        fb = evaluate_field(model, cc, torch.arange(16, device="cuda"), x, cfg, create_graph=False)
        with torch.no_grad():
            out.append(L.descriptors(fb, cfg))
        del fb
    vals.append(torch.cat(out))
v = torch.stack(vals)                                   # draws x codes x K
noise = v.std(dim=0).mean(dim=0)                        # typical per-code spread across draws
signal = v.mean(dim=0).std(dim=0)                       # spread of the true value across codes
scales = torch.tensor(L.DESCRIPTOR_SCALES, device=v.device)
inside = None
print(f"points per shape {cfg.points}; draws {draws}; codes 64")
print("statistic         signal(between codes)  noise(one draw)  signal/noise   noise in scale units")
for i, n in enumerate(L.DESCRIPTOR_NAMES):
    print(f"{n:16s}  {float(signal[i]):10.4f}            {float(noise[i]):8.4f}        {float(signal[i] / noise[i]):6.2f}        {float(noise[i] / scales[i]):6.3f}")
# typical |difference| between two codes that is pure noise, summed as the loss does (mean over statistics)
pure = ((v[0] - v[1]).abs() / scales).mean(dim=1).mean()
real = ((v.mean(0)[:32] - v.mean(0)[32:]).abs() / scales).mean(dim=1).mean()
print(f"statistic gap between two draws of the SAME code: {float(pure):.3f}  |  between different codes (noise-free): {float(real):.3f}")
x = sample_points(cfg, 16, gen, "cuda")
fb = evaluate_field(model, codes[:16], torch.arange(16, device="cuda"), x, cfg, create_graph=False)
print("inside points per shape (mean):", float((fb.f > 0).float().sum(1).mean()))
