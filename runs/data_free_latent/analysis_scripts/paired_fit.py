"""Paired comparison of two unseen-object fits on the same objects.   usage: paired_fit.py <fit tag A> <fit tag B>"""
import json
import sys

import numpy as np

D = "/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/analysis_scripts/followup/"
a, b = (json.load(open(D + t + "_surface.json")) for t in sys.argv[1:3])
assert a["file_ids"] == b["file_ids"]
for key, name in (("distance", "surface distance"), ("iou", "overlap"), ("f02", "F-score at 0.02")):
    d = np.array(b[key]) - np.array(a[key])
    se = d.std(ddof=1) / np.sqrt(len(d))
    print(f"{name}: {sys.argv[2]} minus {sys.argv[1]} = {d.mean():+.4f} +/- {1.96 * se:.4f} (95%); "
          f"{sys.argv[2]} better on {(d < 0).mean() if key == 'distance' else (d > 0).mean():.0%} of objects")
