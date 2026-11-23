"""Thin-wall and gap sites, field errors and probe checks of several runs over the same update windows.

usage: fork_sites_compare.py <run name> [<run name> ...]
A settle run (name ending in a continuation) is simply listed with its own updates.
"""
import json
import sys

R = "/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/"
WINDOWS = [(500, 1000), (1000, 2500), (2500, 4000), (4000, 5000), (5000, 5500), (5500, 6000)]
KEYS = [("graph/wall_sites", "thin-wall sites"), ("graph/gap_sites", "gap sites"), ("mult/thicken", "thicken weight"),
        ("term/consistency", "consistency"), ("term/eikonal", "eikonal")]
for name in sys.argv[1:]:
    rows = [json.loads(line) for line in open(R + name + "/metrics.jsonl")]
    probes = [json.loads(line) for line in open(R + name + "/diagnostics.jsonl")]
    print(name)
    for lo, hi in WINDOWS:
        part = [r for r in rows if lo <= r["step"] < hi]
        if not part:
            continue
        cells = []
        for key, label in KEYS:
            values = [r[key] for r in part if key in r]
            if values:
                cells.append(f"{label} {sum(values) / len(values):.4g}")
        clean = [p for p in probes if lo < p.get("step", -1) <= hi]
        key = next((k for k in ("probe/clean", "topology/clean", "clean") if clean and k in clean[0]), None)
        if key:
            cells.append("probes clean " + " ".join(f"{p[key]:.2f}" for p in clean))
        print(f"   updates {lo:5d} to {hi:5d}: " + "; ".join(cells))
