"""Settings that differ between a reference run and other runs.   usage: config_diff.py <ref run dir> <run dir> [...]"""
import json
import sys

skip = {"steps", "seed", "anchor_path"}
ref = json.load(open(sys.argv[1].rstrip("/") + "/config.json"))
for run in sys.argv[2:]:
    other = json.load(open(run.rstrip("/") + "/config.json"))
    diff = {k: (ref.get(k), other.get(k)) for k in sorted(set(ref) | set(other)) if ref.get(k) != other.get(k) and k not in skip}
    print(run.rstrip("/").split("/")[-1][:40], diff)
