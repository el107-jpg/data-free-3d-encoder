"""Compare unseen-object fits of several models on the objects they have in common.

usage: fit_compare.py <tag> [<tag> ...]      (reads <tag>.json written by heldout_fit.py in the followup folder)
Groups: objects that pass the filled-centre rule (in the first eligible set) and objects that do not.
"""
import json
import sys

import torch

D = "/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/analysis_scripts/followup/"
DS = "/home/eric-luo/Downloads/pointcloud-motion-planning/datasets/"
filled = set(int(i) for i in torch.load(DS + "data_free_eligible_20261111_v1/eligible.pt", map_location="cpu",
                                        weights_only=False)["file_ids"])
runs = {}
for tag in sys.argv[1:]:
    d = json.load(open(D + tag + ".json"))["held_out"]
    runs[tag] = {int(i): v for i, v in zip(d["file_ids"], d["per_object_iou"])}
    print(f"{tag}: {len(runs[tag])} unseen objects, mean overlap {sum(runs[tag].values()) / len(runs[tag]):.3f}")
common = set.intersection(*[set(r) for r in runs.values()])
print(f"\nobjects unseen by every model listed: {len(common)}")
for name, ids in (("all of them", common), ("with a filled centre", common & filled), ("without (thin or hollow-centred)", common - filled)):
    if not ids:
        continue
    print(f"  {name} ({len(ids)} objects)")
    for tag, r in runs.items():
        v = torch.tensor([r[i] for i in sorted(ids)])
        print(f"    {tag:16s} mean {float(v.mean()):.3f}  median {float(v.median()):.3f}  worst {float(v.min()):.3f}  "
              f"share at or above 0.70: {float((v >= 0.7).float().mean()):.2f}")
