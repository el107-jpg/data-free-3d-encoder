"""Warm start a rounded-box decoder from J4 at 5,000 and compare its field with J4's own (CPU)."""
import sys, torch
from arm_planner.data_free import train as T
from arm_planner.data_free.config import DataFreeConfig
from arm_planner.data_free.decoder import ModulatedSphereDecoder
from arm_planner.data_free.latent import sample_sphere

path = sys.argv[1]
state = torch.load(path, map_location="cpu", weights_only=False)
old = DataFreeConfig.from_dict(state["config"])
sphere = ModulatedSphereDecoder(old); sphere.load_state_dict(state["model"]); sphere.eval()
cfg = T.apply_overrides(old, ["base_shape=rounded_box", f"init_model_from={path}", "base_ramp_steps=200"])
trainer = T.Trainer(cfg, device="cpu", tensorboard=False)
box = trainer.model.eval()
g = torch.Generator().manual_seed(0)
z = sample_sphere(16, cfg.latent_dim, g, torch.device("cpu"))
x = torch.rand(16, 4096, 3, generator=g) * 2 - 1
with torch.no_grad():
    a, b = sphere(z, x), box(z, x)
print("max |field difference| at ramp 0:", float((a - b).abs().max()))
box.base_range.fill_(1.0)
with torch.no_grad():
    c = box(z, x)
print("max |field difference| at ramp 1:", float((a - c).abs().max()))
print("anchor codes equal:", torch.equal(trainer.anchor_codes.detach(), state["anchor_codes"]) if "anchor_codes" in state else "none saved")
print("multipliers:", {k: round(v, 2) for k, v in trainer.multipliers.state_dict()["multipliers"].items()})
print("input width", box.input_dim, "base", box.prepare(z[:2]).base.detach())
