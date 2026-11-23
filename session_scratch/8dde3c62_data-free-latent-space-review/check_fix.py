"""After the thin-site fix: wall sites and fragment detection on the blob model (run d, step 10000)."""
import sys

import torch

from arm_planner.data_free import diagnostics as D
from arm_planner.data_free import graph as G
from arm_planner.data_free import graph_losses as GL
from arm_planner.data_free import losses as L
from arm_planner.data_free.field import evaluate_field, sample_points
from arm_planner.data_free.train import Trainer

ckpt = sys.argv[1]
trainer = Trainer.from_checkpoint(ckpt, device="cuda", tensorboard=False)
cfg, model = trainer.cfg, trainer.model.eval()
gen = torch.Generator().manual_seed(123)
walls = gaps = frag = 0
n = 64
for c0 in range(0, n, 8):
    codes = trainer.bank.codes[c0:c0 + 8]
    x = sample_points(cfg, 8, gen, "cuda")
    fb = evaluate_field(model, codes, torch.arange(8, device="cuda"), x, cfg, create_graph=False)
    with torch.no_grad():
        gr = GL.confirm(model, fb.mod, G.build_graph(fb, L.consistency_error(fb), cfg, mode="full"), cfg)
    walls += gr.wall_shape.numel(); gaps += gr.gap_shape.numel()
    frag += int((gr.frag_kind == 0).sum())
print(f"codes {n}: wall sites per code {walls / n:.2f}, gap sites per code {gaps / n:.2f}, "
      f"confirmed detached fragments per code {frag / n:.2f}")
res = D.graph_vs_grid(model, trainer.bank.codes[:32], cfg, resolution=48)
grid = res["grid_counts"]
pieces = grid["pieces"] if isinstance(grid, dict) else torch.tensor([g["pieces"] for g in grid])
print("grid: mean pieces per code", float(torch.as_tensor(pieces).float().mean()),
      "| graph = grid on pieces:", res["pieces_agreement"],
      "| graph fewer:", res["pieces_graph_fewer"], "| graph more:", res["pieces_graph_more"])
