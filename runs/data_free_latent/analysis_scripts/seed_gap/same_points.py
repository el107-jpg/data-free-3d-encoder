"""Trainer's own eikonal and consistency terms for each model on IDENTICAL points (uniform draw and shell draw), 64 codes."""
import copy, sys, torch
from arm_planner.data_free.train import Trainer
from arm_planner.data_free.field import evaluate_field, sample_points
from arm_planner.data_free import losses as L
for item in sys.argv[1:]:
    label, path = item.split("=", 1)
    t = Trainer.from_checkpoint(path, device="cuda", tensorboard=False); m = t.model.eval()
    codes = torch.nn.functional.normalize(torch.randn(64, 64, generator=torch.Generator().manual_seed(31)), dim=-1).cuda()
    for mode in ("uniform", "shells"):
        c = copy.copy(t.cfg); c.point_sampling = mode; e = []; k = []
        for s in range(0, 64, 8):
            x = sample_points(c, 1, torch.Generator(device="cuda").manual_seed(7 + s), "cuda")
            fb = evaluate_field(m, codes[s:s+8], torch.zeros(8, dtype=torch.long, device="cuda"), x, c, create_graph=False)
            e.append(float(L.eikonal(fb))); k.append(float(L.consistency(fb))); del fb
        print(f"{label} on {mode}: eikonal {sum(e)/8:.4f} consistency {sum(k)/8:.4f}")
