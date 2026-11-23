"""Print the logged base statistics of a rounded-box-base run.   usage: base_log.py <run folder> [every]"""
import json
import sys

every = int(sys.argv[2]) if len(sys.argv) > 2 else 50
rows = [json.loads(line) for line in open(sys.argv[1].rstrip("/") + "/metrics.jsonl") if line.strip()]
keys = [k for k in rows[-1] if k.startswith("base/")]
print(keys)
for r in rows[::every] + [rows[-1]]:
    print(r["step"], "  ".join(f"{k.split('/')[1]} {r[k]:.4g}" for k in keys if k in r))
