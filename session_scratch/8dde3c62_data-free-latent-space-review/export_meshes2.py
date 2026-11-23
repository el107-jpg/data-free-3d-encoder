"""Extract surface meshes of chosen shapes from several models for the 3-D viewer."""
import base64
import json

import numpy as np
import torch
from skimage import measure

from arm_planner.data_free import diagnostics as D
from arm_planner.data_free.train import Trainer

T = "/home/eric-luo/.claude/jobs/8dde3c62/tmp"
PICKS = [
    ("Slope loss, 64 shapes per update", f"{T}/df_train/h2_acc64/checkpoint_last.pt", [89, 56, 10]),
    ("Ordering loss, no spread term", f"{T}/k_final.pt", [0, 7, 4]),
    ("Global nearest-neighbour", f"{T}/df_train/r_global_nearest64/checkpoint_last.pt", [26, 2, 0, 3, 5]),
]
res, lim = 96, 0.9
ax = torch.linspace(-lim, lim, res, device="cuda")
grid = torch.stack(torch.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3)
shapes = []
for label, ckpt, picks in PICKS:
    trainer = Trainer.from_checkpoint(ckpt, device="cuda", tensorboard=False)
    model, codes = trainer.model.eval(), trainer.bank.codes
    for i in picks:
        with torch.no_grad():
            f = torch.cat([model(codes[i:i + 1], grid[s:s + 200000][None])[0] for s in range(0, grid.shape[0], 200000)])
        verts, faces, _, _ = measure.marching_cubes(f.reshape(res, res, res).cpu().numpy(), level=0.0)
        verts = verts / (res - 1) * 2 * lim - lim
        q = np.clip(np.round(verts / lim * 32767), -32767, 32767).astype("<i2")
        wide = q.shape[0] > 65535
        idx = faces.astype("<u4" if wide else "<u2")
        coarse = D.grid_topology(lambda p, i=i: model(codes[i:i + 1], p[None])[0], resolution=64, device="cuda")
        fine = D.grid_topology(lambda p, i=i: model(codes[i:i + 1], p[None])[0], resolution=160, device="cuda")
        shapes.append({"run": label, "code": i, "pieces": int(fine["pieces"]), "handles": int(fine["handles"]),
                       "cavities": int(fine["cavities"]), "coarse_handles": int(coarse["handles"]),
                       "tris": int(faces.shape[0]), "wide": bool(wide),
                       "p": base64.b64encode(q.tobytes()).decode(), "i": base64.b64encode(idx.tobytes()).decode()})
        print(label, "code", i, "fine pieces", fine["pieces"], "handles", fine["handles"], "| coarse handles",
              coarse["handles"], "| tris", faces.shape[0])
    del trainer
    torch.cuda.empty_cache()
json.dump({"lim": lim, "shapes": shapes}, open(f"{T}/meshes.json", "w"))
