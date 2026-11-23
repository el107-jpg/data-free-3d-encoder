import json, sys, torch
from arm_planner.data_free import train as T
from arm_planner.data_free.config import DataFreeConfig
from arm_planner.data_free.decoder import ModulatedSphereDecoder
from arm_planner.data_free.latent import sample_sphere
J4 = "/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/J4_d64_nocentre_procedural96_even_code_leak05_beta100_fourier3_5k"
cfg = T.apply_overrides(DataFreeConfig.from_dict(json.load(open(J4 + "/config.json"))),
                        ["base_shape=rounded_box", "base_map=fixed"] + sys.argv[1:])
m = ModulatedSphereDecoder(cfg)
z = sample_sphere(20000, cfg.latent_dim, torch.Generator().manual_seed(0), torch.device("cpu"))
with torch.no_grad():
    base = m.prepare(z).base
b, r = base[:, :3], base[:, 3]
srt = b.sort(dim=1, descending=True).values
long = srt > 2 * r[:, None]                     # an axis clearly longer than the rounding
n_long = long.sum(1)
q = lambda v: [round(float(x), 3) for x in torch.quantile(v, torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95]))]
print("r quantiles 5/25/50/75/95%:", q(r))
print("longest half-extent b+r:", q(srt[:, 0] + r), " thinnest half-extent:", q(srt[:, 2] + r))
print("long axes (b > 2r): 0 %.2f  1 (rod) %.2f  2 (sheet) %.2f  3 (box) %.2f" % tuple(float((n_long == k).float().mean()) for k in range(4)))
thin = (srt[:, 2] + r) < 0.12
print("thin (thinnest half-extent < 0.12): %.2f; aspect longest/thinnest median %.1f, 95%% %.1f" % (
    float(thin.float().mean()), *[float(x) for x in torch.quantile((srt[:, 0] + r) / (srt[:, 2] + r), torch.tensor([0.5, 0.95]))]))
# anchors
d = torch.load(cfg.anchor_path, map_location="cpu", weights_only=False)
grid = d["grid"].float()
fit, err = T.fit_rounded_boxes(grid, torch.Generator().manual_seed(1))
inside = grid > 0
print("anchor fit error (weighted mean |field diff|): median %.3f, 90%% %.3f, max %.3f" % (
    float(err.median()), float(torch.quantile(err, 0.9)), float(err.max())))
x = torch.linspace(-1, 1, 96)
X = torch.stack(torch.meshgrid(x, x, x, indexing="ij"), -1).reshape(1, -1, 3)
from arm_planner.data_free.decoder import rounded_box_field
ious = []
for s in range(0, 96, 8):
    f = rounded_box_field(X.expand(min(8, 96 - s), -1, -1), fit[s:s + 8])
    a, t = f > 0, inside[s:s + 8].reshape(min(8, 96 - s), -1)
    ious.append((a & t).sum(1) / (a | t).sum(1))
iou = torch.cat(ious)
print("anchor box-fit volume IoU: median %.2f, 10%% %.2f" % (float(iou.median()), float(torch.quantile(iou, 0.1))))
corner = fit[:, :3].norm(dim=-1) + fit[:, 3]
print("anchor fits: r", q(fit[:, 3]), " corner reach", q(corner), " over reach limit %.2f" % float((corner > cfg.max_reach()).float().mean()))
s = m.scores_for_base(fit)
print("scores needed: |s| max per anchor quantiles", q(s.abs().max(1).values), " clamped share %.2f" % float((s.abs() >= 3.5 - 1e-6).any(1).float().mean()))
