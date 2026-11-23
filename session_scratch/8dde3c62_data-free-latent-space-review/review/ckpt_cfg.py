import sys, json, torch
ROOT = "/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/"
from arm_planner.data_free.config import DataFreeConfig
base = DataFreeConfig().to_dict()
for run in sys.argv[1:]:
    st = torch.load(f"{ROOT}{run}/checkpoint_last.pt", map_location="cpu", weights_only=False)
    c = st["config"]
    diff = {}
    for k, v in c.items():
        b = base.get(k, "NEW")
        if isinstance(v, dict):
            sub = {kk: vv for kk, vv in v.items() if (list(b.get(kk)) if isinstance(b.get(kk), tuple) else b.get(kk)) != (list(vv) if isinstance(vv, tuple) else vv)}
            if sub:
                diff[k] = sub
        elif v != b:
            diff[k] = v
    print("==", run, "step", st["step"])
    print("   ", diff)
    print("    multipliers:", {k: round(v, 1) for k, v in st["multipliers"]["multipliers"].items()})
    print("    ema:", {k: float(f"{v:.2e}") for k, v in st["multipliers"]["ema"].items()})
    # Adam state sanity: lr
    print("    lr now:", st["optimizer"]["param_groups"][0]["lr"])
