"""Why did handles form without 'widen' reacting? Gap sites vs grid handles on a saved model."""
import sys

import torch

from arm_planner.data_free import diagnostics as D
from arm_planner.data_free import graph as G
from arm_planner.data_free import graph_losses as GL
from arm_planner.data_free import losses as L
from arm_planner.data_free.field import evaluate_field, sample_points
from arm_planner.data_free.train import Trainer

trainer = Trainer.from_checkpoint(sys.argv[1], device="cuda", tensorboard=False)
cfg, model = trainer.cfg, trainer.model.eval()
t = float(cfg.min_feature)
n = 32
codes = trainer.bank.codes[:n]
grid = D.decoder_grid_topology(model, codes, cfg, resolution=64)
handles = torch.tensor([g["handles"] for g in grid])
pieces = torch.tensor([g["pieces"] for g in grid])
print(f"grid (res 64): codes with handles {int((handles > 0).sum())}/{n}, total handles {int(handles.sum())}, "
      f"codes with >1 piece {int((pieces > 1).sum())}")

gen = torch.Generator().manual_seed(5)
tot = {"gap": 0, "gap_active": 0, "wall": 0, "wall_active": 0}
per_code_gap = torch.zeros(n)
lens, mids = [], []
for rep in range(3):                                  # three independent sample draws per code
    for c0 in range(0, n, 8):
        cc = codes[c0:c0 + 8]
        x = sample_points(cfg, 8, gen, "cuda")
        fb = evaluate_field(model, cc, torch.arange(8, device="cuda"), x, cfg, create_graph=False)
        with torch.no_grad():
            gr = GL.confirm(model, fb.mod, G.build_graph(fb, L.consistency_error(fb), cfg, mode="full"), cfg)
            fg = model(cc[gr.gap_shape], gr.gap_mid[:, None, :])[:, 0] if gr.gap_shape.numel() else torch.zeros(0, device="cuda")
            fw = model(cc[gr.wall_shape], gr.wall_mid[:, None, :])[:, 0] if gr.wall_shape.numel() else torch.zeros(0, device="cuda")
        tot["gap"] += gr.gap_shape.numel(); tot["gap_active"] += int((fg < -0.25 * gr.gap_len).sum())
        tot["wall"] += gr.wall_shape.numel(); tot["wall_active"] += int((fw > 0.25 * gr.wall_len).sum())
        per_code_gap += torch.bincount(gr.gap_shape.cpu() + c0, minlength=n).float()
        lens.append(gr.gap_len.cpu()); mids.append(fg.cpu())
lens, mids = torch.cat(lens), torch.cat(mids)
print("per draw of 32 codes:", {k: round(v / 3, 1) for k, v in tot.items()})
if lens.numel():
    print(f"gap length median {float(lens.median()):.3f} (t = {t}), midpoint value median {float(mids.median()):.4f}")
with_h = handles > 0
print(f"mean gap sites per draw: codes with handles {float(per_code_gap[with_h].mean() / 3 if with_h.any() else 0):.2f}, "
      f"codes without {float(per_code_gap[~with_h].mean() / 3):.2f}")
# how narrow are the necks of the existing handles? (cannot be widened once formed: a handle is closed)
print("note: an existing handle is solid all the way round; widen acts only on open gaps")
