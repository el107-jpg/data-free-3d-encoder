"""Target sets for the fitting test on the capped-follow model.

set A: the 86 objects eligible under the filled-centre rule that were never anchors (same set as yesterday's test).
set B: 64 random objects that are eligible only WITHOUT the filled-centre rule (thin, flat or off-centre shapes
       the model was never built to make), none of them ever an anchor of this model.
"""
import torch

D = "/home/eric-luo/Downloads/pointcloud-motion-planning/datasets/"
v1 = torch.load(D + "data_free_eligible_20261111_v1/eligible.pt", map_location="cpu", weights_only=False)
v3 = torch.load(D + "data_free_anchors_20261112_v3/eligible.pt", map_location="cpu", weights_only=False)
print("keys:", sorted(v1.keys()))
ids1 = [int(i) for i in v1["file_ids"]]
ids3 = [int(i) for i in v3["file_ids"]]
only3 = [k for k, i in enumerate(ids3) if i not in set(ids1)]
gen = torch.Generator().manual_seed(4)
pick = [only3[int(j)] for j in torch.randperm(len(only3), generator=gen)[:64]]
out = {k: v for k, v in v3.items() if k not in ("grid", "file_ids", "chosen")}
out["grid"] = v3["grid"][pick]
out["file_ids"] = [ids3[k] for k in pick]
out["chosen"] = [False] * len(pick)
if "names" in v3:
    out["names"] = [v3["names"][k] for k in pick]
torch.save(out, "/home/eric-luo/.claude/jobs/8dde3c62/tmp/fit_set_B.pt")
print(f"v1 eligible {len(ids1)}; v3 eligible {len(ids3)}; eligible only without the centre rule {len(only3)}; set B {len(pick)} objects")
print("grid shapes:", tuple(v1["grid"].shape), tuple(out["grid"].shape))
