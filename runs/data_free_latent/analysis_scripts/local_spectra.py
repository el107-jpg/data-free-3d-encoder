"""Side-by-side local spectra (surface motion per latent direction at each code, pose removed).

usage: local_spectra.py <tag> [<tag> ...]   (reads shell_variance_<tag>.json written by shell_variance.py)
"""
import json
import sys

import numpy as np

TMP = "/home/eric-luo/.claude/jobs/8dde3c62/tmp/"
print("tag                 total   eff   >1e-2  >1e-3  >1e-4 | RMS surface motion per unit chord, directions 1..16")
for tag in sys.argv[1:]:
    d = json.load(open(f"{TMP}shell_variance_{tag}.json"))
    v = np.array(d["local_pose_removed_variances"])
    s = d["local_pose_removed"]
    print(f'{tag:18s} {s["total"]:6.3f}  {s["effective_directions"]:4.2f}  {int((v > 1e-2).sum()):5d}  {int((v > 1e-3).sum()):5d}  '
          f'{int((v > 1e-4).sum()):5d} | ' + " ".join(f"{x:.3f}" for x in np.sqrt(v[:16])))
