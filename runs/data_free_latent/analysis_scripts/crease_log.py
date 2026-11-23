"""Print the crease term's logged values (and a few others) from a run's metrics.

usage: crease_log.py <run folder> [every]
"""
import json
import sys

every = int(sys.argv[2]) if len(sys.argv) > 2 else 1
rows = [json.loads(line) for line in open(sys.argv[1].rstrip("/") + "/metrics.jsonl") if line.strip()]
keys = ("crease/term", "crease/used", "crease/turn", "crease/flat", "crease/sharp", "loss", "grad_norm",
        "term/consistency", "term/eikonal", "mult/consistency", "diversity", "anchor/iou", "time_ms/losses", "time_ms/backward")
for r in rows[::every] + ([rows[-1]] if (len(rows) - 1) % every else []):
    print(r["step"], "  ".join(f"{k.split('/')[-1]} {r[k]:.3g}" for k in keys if k in r))
