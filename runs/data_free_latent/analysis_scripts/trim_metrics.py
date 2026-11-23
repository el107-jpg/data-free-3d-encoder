"""Trim a run's metrics.jsonl and diagnostics.jsonl back to a checkpoint's step before resuming after a crash.

usage: trim_metrics.py <run folder> <checkpoint step>
Keeps lines that parse and whose step is below the checkpoint step; the originals are kept as *.before_crash.
"""
import json
import os
import shutil
import sys

folder, step = sys.argv[1].rstrip("/"), int(sys.argv[2])
for name in ("metrics.jsonl", "diagnostics.jsonl"):
    path = os.path.join(folder, name)
    if not os.path.exists(path):
        continue
    shutil.copy(path, path + ".before_crash")
    kept, dropped = [], 0
    with open(path, "rb") as handle:
        for raw in handle:
            try:
                record = json.loads(raw)
            except Exception:
                dropped += 1
                continue
            if int(record.get("step", -1)) < step:
                kept.append(json.dumps(record))
            else:
                dropped += 1
    with open(path, "w") as handle:
        handle.write("\n".join(kept) + "\n")
    print(f"{name}: kept {len(kept)}, dropped {dropped}")
