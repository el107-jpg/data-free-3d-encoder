"""On B12 resumed with bends/tapers + base repulsion (weight 1): size of the repulsion's gradient on the base
head versus the main diversity term's and the whole loss's, for one real update (CPU)."""
import torch
from arm_planner.data_free import losses as L
from arm_planner.data_free.train import Trainer
B12 = "/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/B12_B11_cont_corrcost50_to9k/checkpoint_last.pt"
t = Trainer.from_checkpoint(B12, device="cpu", tensorboard=False, overrides=[
    "base_deform=True", "base_deform_init_std=1e-3", "base_repel_weight=1.0", "base_spread_target=0", "base_spread_target_r=0"])
head = list(t.model.base_head_mlp.parameters())
acc = {}
def take(name, value):
    if torch.is_tensor(value) and value.requires_grad:
        g = torch.autograd.grad(value, head, retain_graph=True, allow_unused=True)
        acc[name] = acc.get(name, 0.0) + sum(float(x.square().sum()) for x in g if x is not None) ** 0.5
br, comb = L.base_repel, t.multipliers.combine
def br2(*a, **k):
    loss, info = br(*a, **k); take("repel", loss); acc["repel_value"] = acc.get("repel_value", 0.0) + float(loss); return loss, info
def comb2(objective, constraints):
    take("variety", objective)
    return comb(objective, constraints)
L.base_repel, t.multipliers.combine = br2, comb2
r = t.step()
print({k: round(v, 3) for k, v in acc.items()}, "loss", round(r["loss"], 2), "repel term", round(r.get("base_repel/term", float("nan")), 3), "nearest", round(r.get("base_repel/nearest", float("nan")), 3))
