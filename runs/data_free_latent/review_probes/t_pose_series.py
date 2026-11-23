import json
import numpy as np
ROOT = "/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/t_batchwide128/"
rows = [json.loads(l) for l in open(ROOT + "metrics.jsonl")]
print("rows", len(rows), "last step", rows[-1]["step"], "lr last", rows[-1]["lr"])
steps = np.array([r["step"] for r in rows])
for k in ("term/pose_centroid", "term/pose_scale", "term/pose_axes", "term/consistency", "mult/consistency", "div/nearest_distance", "graph/wall_sites", "term/thicken", "lr"):
    v = np.array([r[k] for r in rows], dtype=float)
    out = []
    for a in range(2300, int(steps.max()) + 1, 50):
        m = (steps >= a) & (steps < a + 50)
        if m.any():
            out.append(f"{a}:{v[m].mean():.3g}")
    print(k, " ".join(out))
d = [json.loads(l) for l in open(ROOT + "diagnostics.jsonl")]
print("diagnostics rows (step, centroid_offset_mean, centroid_sq_mean, trace_mean, trace_rel_error_rms, live, eff, clean):")
for r in d[-8:]:
    print(int(r["step"]), round(r["pose/centroid_offset_mean"], 4), f"{r['pose/centroid_sq_mean']:.2e}", round(r["pose/trace_mean"], 4), round(r["pose/trace_rel_error_rms"], 3),
          r.get("dead/live_mean"), round(r.get("dead/effective_directions_mean", float('nan')), 2), r.get("grid/clean_fraction"))
