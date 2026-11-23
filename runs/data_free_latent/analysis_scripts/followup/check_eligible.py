"""Does the rebuilt eligible set reproduce the original 64 anchors exactly?"""
import json

import torch

D = "/home/eric-luo/Downloads/pointcloud-motion-planning/datasets/"
old = torch.load(D + "data_free_anchors_20261111_v1/anchors.pt", weights_only=False)
new = torch.load(D + "data_free_eligible_20261111_v1/anchors.pt", weights_only=False)
elig = torch.load(D + "data_free_eligible_20261111_v1/eligible.pt", weights_only=False)
print("same chosen file ids, same order:", old["file_ids"] == new["file_ids"])
print("largest grid difference between the two builds:", float((old["grid"].float() - new["grid"].float()).abs().max()))
chosen = [f for f, c in zip(elig["file_ids"], elig["chosen"]) if c]
print("eligible:", len(elig["file_ids"]), "chosen:", sum(elig["chosen"]), "held out:", len(elig["file_ids"]) - sum(elig["chosen"]),
      "| chosen set equals the anchors:", sorted(chosen) == sorted(old["file_ids"]))
