import math, sys, torch
sys.path.insert(0, "tests")
import test_data_free_losses as T
from arm_planner.data_free import losses as L
from arm_planner.data_free.config import DataFreeConfig
base = T._smooth_blob
for move in [dict(shift=(0.02, -0.01, 0.015)), dict(shift=(0.01, -0.005, 0.0075)), dict(angle=0.05), dict(scale=1.04), dict(scale=1.02),
             dict(shift=(0.01, 0.01, 0.0), angle=0.03, scale=0.97)]:
    fb = T._pair_fb(base, T._moved(base, **move))
    cfg = DataFreeConfig(); cfg.diversity_pose_invariant = False
    raw = float(L.diversity(fb, cfg)[1]["mean_shape_distance"])
    cfg.diversity_pose_invariant = True
    inv = float(L.diversity(fb, cfg)[1]["mean_shape_distance"])
    print(move, f"raw {raw:.5f} after {inv:.5f} removed {1 - inv / raw:.3f}")
