import json, sys
import numpy as np
ROOT = "/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/"
run = sys.argv[1]; lo = int(sys.argv[2]); hi = int(sys.argv[3])
rows = [json.loads(l) for l in open(f"{ROOT}{run}/metrics.jsonl")]
rows = [r for r in rows if lo <= r["step"] < hi]
print(run, "steps", rows[0]["step"], "to", rows[-1]["step"], "n", len(rows))
for k in ("term/consistency", "mult/consistency", "term/pose_scale", "term/pose_centroid", "term/thicken", "mult/thicken", "graph/wall_sites", "graph/gap_sites",
          "graph/fragments_kind0", "graph/fragments_kind1", "graph/confirm_dropped_kind0", "graph/extra_pieces", "div/nearest_distance", "e", "grad_norm", "term/eikonal", "term/barrier", "mult/barrier"):
    v = np.array([r.get(k, np.nan) for r in rows], dtype=float)
    print(f"{k:28s} mean {np.nanmean(v):10.4g}  p5 {np.nanpercentile(v,5):10.4g}  p50 {np.nanpercentile(v,50):10.4g}  p95 {np.nanpercentile(v,95):10.4g}  last5 {np.array2string(v[-5:], precision=4)}")
# controller cycle: autocorrelation of the consistency multiplier
m = np.array([r["mult/consistency"] for r in rows]); c = np.array([r["term/consistency"] for r in rows])
print("consistency multiplier every 25 steps:", np.array2string(m[::25], precision=0))
print("consistency term (25-step means):", np.array2string(np.array([c[i:i+25].mean() for i in range(0, len(c), 25)]), precision=4))
w = np.array([r["graph/wall_sites"] for r in rows])
print("wall sites (100-step means):", np.array2string(np.array([w[i:i+100].mean() for i in range(0, len(w), 100)]), precision=1))
