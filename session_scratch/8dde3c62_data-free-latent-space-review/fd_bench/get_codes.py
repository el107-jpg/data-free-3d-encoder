"""Encode 32 validation objects with a trained d32 run; save the codes (read-only on the repo)."""
import json, sys, time
from pathlib import Path
import numpy as np, torch
from arm_planner.object_geometry.data import _seed
from arm_planner.object_geometry.full_library import load_object
from arm_planner.object_geometry.pointnet import CLOUD_VERSION, farthest_point_indices, surface_cloud
from arm_planner.object_geometry.sdf_recipe import MeshSDF, RecipeConfig

run = Path(sys.argv[1]); out = Path(sys.argv[2])
saved = torch.load(run / 'latest.pt', map_location='cpu', weights_only=False)
args = json.loads((run / 'config.json').read_text())['args']
lib = Path(args['library'])
entries = [e for e in json.loads((lib / 'manifest.json').read_text())['objects'] if e['split'] == 'validation']
rng = np.random.default_rng(0); chosen = [entries[i] for i in rng.choice(len(entries), 32, replace=False)]
model = MeshSDF(RecipeConfig(**saved['model_config'])).cuda().eval().requires_grad_(False); model.load_state_dict(saved['model'])
codes = []; t = time.time()
with torch.inference_mode():
    for e in chosen:
        obj, _, center, scale, q, field = load_object(lib, e)
        cloud = surface_cloud(obj, center, scale, args['point_count'], _seed(CLOUD_VERSION, args['seed'], e['id'], 0))
        anchors = farthest_point_indices(cloud[:, :3].numpy(), model.config.latent_set_tokens)
        codes.append(model.encode({'points': cloud[None].cuda(), 'point_anchor_indices': anchors[None].cuda()})[0].float().cpu())
codes = torch.stack(codes); n = codes.norm(dim=1)
print('codes', tuple(codes.shape), 'norm min/med/max', n.min().item(), n.median().item(), n.max().item(), 'sec', time.time() - t)
print('groups', [e.get('mixture_group') for e in chosen])
torch.save({'codes': codes, 'ids': [e['id'] for e in chosen], 'groups': [e.get('mixture_group') for e in chosen],
            'step': int(saved['step']), 'run': str(run)}, out)
