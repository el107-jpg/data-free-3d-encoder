"""Extract surface meshes of generated shapes (some with handles) for the 3-D viewer."""
import base64
import json
import sys

import numpy as np
import torch
from skimage import measure

from arm_planner.data_free import diagnostics as D
from arm_planner.data_free.train import Trainer

ckpt, out = sys.argv[1], sys.argv[2]
trainer = Trainer.from_checkpoint(ckpt, device="cuda", tensorboard=False)
cfg, model = trainer.cfg, trainer.model.eval()
codes = trainer.bank.codes[:96]
topo = D.decoder_grid_topology(model, codes, cfg, resolution=64)
handles = [int(t["handles"]) for t in topo]
with_h = [i for i, h in enumerate(handles) if h > 0][:5]
without = [i for i, h in enumerate(handles) if h == 0 and topo[i]["pieces"] == 1][:8 - len(with_h)]
chosen = with_h + without
print("codes with handles among 96:", sum(h > 0 for h in handles), "| chosen", chosen)

res, lim = 96, 0.9
ax = torch.linspace(-lim, lim, res, device="cuda")
grid = torch.stack(torch.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3)
shapes = []
for i in chosen:
    with torch.no_grad():
        f = torch.cat([model(codes[i:i + 1], grid[s:s + 200000][None])[0] for s in range(0, grid.shape[0], 200000)])
    vol = f.reshape(res, res, res).cpu().numpy()
    verts, faces, _, _ = measure.marching_cubes(vol, level=0.0)
    verts = verts / (res - 1) * 2 * lim - lim                         # back to model coordinates
    q = np.clip(np.round(verts / lim * 32767), -32767, 32767).astype("<i2")
    fine = D.grid_topology(lambda p, i=i: model(codes[i:i + 1], p[None])[0], resolution=96, device="cuda")
    shapes.append({"code": i, "pieces": int(fine["pieces"]), "handles": int(fine["handles"]),
                   "cavities": int(fine["cavities"]), "verts": int(q.shape[0]), "tris": int(faces.shape[0]),
                   "p": base64.b64encode(q.tobytes()).decode(), "i": base64.b64encode(faces.astype("<u4").tobytes()).decode()})
    print("code", i, "pieces", fine["pieces"], "handles", fine["handles"], "cavities", fine["cavities"], "tris", faces.shape[0])
json.dump({"lim": lim, "step": trainer.step_count, "shapes": shapes}, open(out, "w"))
print("wrote", out)
