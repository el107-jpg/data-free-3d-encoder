"""Why did the learned rounded-box base stay a sphere? Read the base head from a checkpoint (no model class needed).

usage: base_head_check.py <checkpoint of a base_shape=rounded_box run> [<diagnostics.jsonl>]

For 2,000 random codes: the head's raw outputs u (three extents, one radius), the soft ramp's value and slope
at those outputs (extent p = tau (softplus(u / tau) - ln 2), slope sigmoid(u / tau), tau = 0.01), and the
resulting base. A slope of zero means no gradient can reach the head through that extent.
"""
import json
import math
import sys

import torch
import torch.nn.functional as F

state = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
cfg, model = state["config"], state["model"]
beta, tau = float(cfg["softplus_beta"]), 0.01
gen = torch.Generator().manual_seed(1)
z = F.normalize(torch.randn(2000, int(cfg["latent_dim"]), generator=gen), dim=-1)
h = z
k = 0
while f"mapping.{k}.weight" in model:
    h = F.softplus(h @ model[f"mapping.{k}.weight"].T + model[f"mapping.{k}.bias"], beta=beta)
    k += 2
u = float(cfg.get("base_speed", 1.0)) * (h @ model["base_head.weight"].T + model["base_head.bias"])
slope = torch.sigmoid(u[:, :3] / tau)
p = tau * (F.softplus(u[:, :3] / tau) - math.log(2.0))
print(f"update {state['step']}; side-network features: typical size {float(h.abs().mean()):.2f} over {h.shape[1]} features")
print(f"head weights: typical size {float(model['base_head.weight'].abs().mean()):.4f}; bias {[round(float(v), 3) for v in model['base_head.bias']]}")
for i, name in enumerate(("extent step 1 (longest)", "extent step 2", "extent step 3 (shortest)")):
    print(f"  {name:26s} raw output: mean {float(u[:, i].mean()):+.3f}, range {float(u[:, i].min()):+.3f} to {float(u[:, i].max()):+.3f}; "
          f"ramp value {float(p[:, i].mean()):+.4f}; ramp slope mean {float(slope[:, i].mean()):.2e}, largest {float(slope[:, i].max()):.2e}; "
          f"codes with slope over 0.01: {int((slope[:, i] > 0.01).sum())} of 2000")
print(f"  radius output: mean {float(u[:, 3].mean()):+.3f}, range {float(u[:, 3].min()):+.3f} to {float(u[:, 3].max()):+.3f}")
print(f"  for scale: one Adam step of {float(cfg['lr']):.2e} on every head weight can move a raw output by up to "
      f"{float(cfg['lr']) * float(h.abs().sum(1).mean()):.3f} (the ramp's temperature is {tau})")
if len(sys.argv) > 2:
    rows = [json.loads(line) for line in open(sys.argv[2]) if line.strip()]
    keys = [k for k in rows[-1] if k.startswith("base/")]
    for r in rows:
        print(f"  check at update {int(r['step'])}: " + ", ".join(f"{k.split('/')[1]} {r[k]:.4g}" for k in keys if k in r))
