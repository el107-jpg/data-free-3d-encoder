"""Side-by-side table of evalrun.py results, plus training terms from each run's log.

usage: table.py <tag>[=<run folder>[:lo:hi]] ...
With a run folder, the mean nearest-neighbour distance, thin-wall sites, consistency term/multiplier
range and anchor overlap over updates [lo, hi) of that run's metrics are added.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/"
print(f'{"tag":26s} upd   dim  total   eff  >.02 >.01 >.005 | 11th  12th  13th  14th  15th  16th | clean/128 (res96, fine) | centroid  trace  | anchors iou (min)')
for item in sys.argv[1:]:
    tag, _, run = item.partition("=")
    d = json.load(open(os.path.join(HERE, tag + ".json")))
    s, t, p = d["spectrum"], d["topology"], d["pose"]
    r = s["rms"] + [0.0] * 16
    a = d.get("anchors")
    print(f'{tag:26s} {d["update"]:5d} {d["latent_dim"]:3d}  {s["total_variance"]:.3f}  {s["effective_directions"]:4.2f}  '
          f'{s["above_0.02"]:3d}  {s["above_0.01"]:3d}  {s["above_0.005"]:3d}  | ' + " ".join(f"{v:.3f}" for v in r[10:16])
          + f' | {t["clean_at_96"]:3d}, {t["clean_on_fine_grids"]:3d}            | {p["centroid_offset_mean"]:.4f}  {p["trace_mean"]:.4f} | '
          + (f'{a["iou_mean"]:.2f} ({a["iou_min"]:.2f})' if a else "n/a"))
    if run:
        name, lo, hi = (run.split(":") + ["0", "999999999"])[:3]
        rows = {}
        for line in open(ROOT + name + "/metrics.jsonl"):
            m = json.loads(line)
            if int(lo) <= m["step"] < int(hi):
                rows[m["step"]] = m
        rows = list(rows.values())
        if rows:
            mean = lambda k: sum(x.get(k, 0.0) for x in rows) / len(rows)
            lo_ = lambda k: min(x.get(k, 0.0) for x in rows)
            hi_ = lambda k: max(x.get(k, 0.0) for x in rows)
            print(f'    {name} updates {rows[0]["step"]}..{rows[-1]["step"]}: nearest {mean("div/nearest_distance"):.4f}  '
                  f'consistency {mean("term/consistency"):.5f} ({lo_("term/consistency"):.5f}..{hi_("term/consistency"):.5f})  '
                  f'multiplier {mean("mult/consistency"):.1f} ({lo_("mult/consistency"):.1f}..{hi_("mult/consistency"):.1f})  '
                  f'wall sites {mean("graph/wall_sites"):.2f}  gap sites {mean("graph/gap_sites"):.2f}  thicken mult {mean("mult/thicken"):.1f}  '
                  f'anchor overlap in training {mean("anchor/iou"):.3f}  skipped {sum(x.get("skipped", 0) for x in rows)}')
