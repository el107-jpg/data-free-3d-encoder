"""Settle stage: consistency term and multiplier over windows of a run's log.

usage: settle_log.py <run folder> <lo> <hi> [<lo> <hi> ...]
"""
import json
import sys

ROOT = "/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/"
run = sys.argv[1]
rows = {}
for line in open(ROOT + run + "/metrics.jsonl"):
    m = json.loads(line)
    rows[m["step"]] = m
tol = json.load(open(ROOT + run + "/config.json"))["tolerances"]["consistency"] if False else None
for lo, hi in zip(sys.argv[2::2], sys.argv[3::2]):
    rs = [rows[s] for s in sorted(rows) if int(lo) <= s < int(hi)]
    if not rs:
        continue
    def stat(k):
        v = [r.get(k, 0.0) for r in rs]
        return sum(v) / len(v), min(v), max(v)
    c, m = stat("term/consistency"), stat("mult/consistency")
    print(f"{run} updates {rs[0]['step']}..{rs[-1]['step']} ({len(rs)}): consistency mean {c[0]:.5f} (min {c[1]:.5f}, max {c[2]:.5f}); "
          f"multiplier mean {m[0]:.1f} (min {m[1]:.1f}, max {m[2]:.1f}, max/min {m[2] / max(m[1], 1e-9):.2f}); "
          f"lr {rs[-1]['lr']:.3g}; nearest {stat('div/nearest_distance')[0]:.4f}; pose_scale mult {stat('mult/pose_scale')[0]:.1f}; "
          f"pose_centroid mult {stat('mult/pose_centroid')[0]:.1f}; wall sites {stat('graph/wall_sites')[0]:.2f}; "
          f"grad norm {stat('grad_norm')[0]:.1f}; skipped {sum(r.get('skipped', 0) for r in rs)}")
