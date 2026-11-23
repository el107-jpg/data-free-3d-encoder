"""Save the settings stored inside each run's last checkpoint as JSON (weights are not read or copied).

usage: extract_configs.py <code root to import arm_planner.data_free from> [run ...]
"""
import json, os, sys
sys.path.insert(0, sys.argv[1])
import torch
ROOT = "/home/eric-luo/Downloads/pointcloud-motion-planning"
SRC = ROOT + "/runs/data_free_latent"
OUT = ROOT + "/data_free_latent_transfer/run_configs_from_checkpoints"
os.makedirs(OUT, exist_ok=True)
runs = sys.argv[2:] or sorted(d for d in os.listdir(SRC) if os.path.isfile(f"{SRC}/{d}/checkpoint_last.pt"))

def plain(v):
    if isinstance(v, dict):
        return {str(k): plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [plain(x) for x in v]
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)

failed = []
for run in runs:
    try:
        st = torch.load(f"{SRC}/{run}/checkpoint_last.pt", map_location="cpu", weights_only=False, mmap=True)
        cfg = st["config"]
        if not isinstance(cfg, dict):
            cfg = cfg.to_dict() if hasattr(cfg, "to_dict") else vars(cfg)
        rec = {"run": run, "source": "checkpoint_last.pt", "update": int(st.get("step", -1)), "config": plain(cfg)}
        with open(f"{OUT}/{run}.json", "w") as f:
            json.dump(rec, f, indent=1, sort_keys=True)
        del st
    except Exception as e:  # noqa: BLE001
        failed.append(run); print("FAILED", run, type(e).__name__, str(e)[:160])
print(f"{len(runs) - len(failed)} written, {len(failed)} failed")
print("FAILED_LIST " + " ".join(failed))
