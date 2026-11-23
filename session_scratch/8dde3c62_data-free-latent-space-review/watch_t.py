"""Wake-up watch for the long run: exits (printing why) at a milestone, on trouble, or when the trainer stops.

usage: watch_t.py <milestone update>
Trouble = a skipped update, a non-finite loss, a corrective multiplier above 50 (wind-up),
or two diagnostics in a row with fewer than 85% of probes clean.
"""
import json
import math
import subprocess
import sys
import time

T = "/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/t_batchwide128/"
PAT = "train_data_free_latent.py --output-dir .*t_batchwide128 --steps 20000"
milestone = int(sys.argv[1])
CORRECTIVE = ("bridge", "delete", "fill", "widen", "thicken")


def tail_rows(path, size=600_000):
    with open(path, "rb") as fh:
        fh.seek(0, 2)
        end = fh.tell()
        fh.seek(max(0, end - size))
        text = fh.read().decode("utf-8", "ignore")
    out = []
    for line in text.splitlines()[1:]:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


while True:
    alive = subprocess.run(["pgrep", "-f", PAT], capture_output=True).returncode == 0
    rows = tail_rows(T + "metrics.jsonl")[-200:]
    diag = tail_rows(T + "diagnostics.jsonl", 60_000)[-2:]
    step = rows[-1]["step"] if rows else -1
    why = []
    if not alive:
        why.append("trainer process is not running")
    if rows:
        if sum(r.get("skipped", 0) for r in rows) > 0:
            why.append("skipped updates in the last 200")
        if any(not math.isfinite(r.get("loss", 0.0)) for r in rows):
            why.append("non-finite loss")
        for name in CORRECTIVE:
            mean = sum(r.get("mult/" + name, 1.0) for r in rows[-100:]) / len(rows[-100:])
            if mean > 50:
                why.append(f"{name} multiplier mean {mean:.0f} over the last 100 updates")
    if len(diag) == 2 and all(d.get("grid/clean_fraction", 1.0) < 0.85 for d in diag):
        why.append("fewer than 85% of probes clean in two diagnostics in a row")
    if step >= milestone:
        why.append(f"milestone {milestone} reached")
    if why:
        print(f"update {step}: " + "; ".join(why))
        break
    time.sleep(30)
