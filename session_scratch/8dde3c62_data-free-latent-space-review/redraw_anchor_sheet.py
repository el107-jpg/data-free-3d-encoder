"""Redraw the contact sheet of a procedural anchor set with a given title.

usage: redraw_anchor_sheet.py <anchor folder> <out.png> <title line>
"""
import json
import os
import sys

import torch

sys.path.insert(0, "scripts")
import build_data_free_procedural_anchors as B  # noqa: E402

folder, out, lead = sys.argv[1], sys.argv[2], sys.argv[3]
records = json.load(open(os.path.join(folder, "anchors.json")))["anchors"]
grids = torch.load(os.path.join(folder, "anchors.pt"), map_location="cpu", weights_only=False)["grid"].numpy()
B.contact_sheet(records, grids, out,
                lead + ("   (* = scaled down to fit inside the outer radius)\n" if any(r.get("scaled_down_to_fit") for r in records) else "\n") +
                "Under each: local character (rod / sheet / blob shares); mean surface crossings of rays from the "
                "centre (1 = star-shaped); fold; solidity; thickness; branches; nearest gap")
thick = [r["descriptor"]["thickness"] for r in records]
print("saved", out, "|", len(records), "anchors; thickness of anchors 0-63:", round(min(thick[:64]), 2), "to", round(max(thick[:64]), 2),
      "| anchors 64-79:", round(min(thick[64:80]), 2), "to", round(max(thick[64:80]), 2),
      "| anchors 80-95:", round(min(thick[80:]), 2), "to", round(max(thick[80:]), 2))
