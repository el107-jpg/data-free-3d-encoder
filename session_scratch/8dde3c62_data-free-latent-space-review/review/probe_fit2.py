"""CPU-only, round two. Fit codes of a data-free decoder to analytic targets in the decoder's pose convention.
Target sets: 'solids' (the six of round one) and 'bar' (analytic stand-ins for the kinds of shape the agreed bar
uses as novel targets: thin sheet, plate with hole, open shelf, shelf with back, ring, bent sheet).
Half of the fitting points are taken near the target surface so thin targets give a signal. Indicative only."""
import sys, time
import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(6)
ROOT = "/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/"
run, ckpt = (sys.argv[1].split(":") + ["checkpoint_last.pt"])[:2]
which = sys.argv[2]; steps = int(sys.argv[3])
from arm_planner.data_free.config import DataFreeConfig
from arm_planner.data_free.decoder import ModulatedSphereDecoder
from arm_planner.data_free import analytic_shapes as A

state = torch.load(f"{ROOT}{run}/{ckpt}", map_location="cpu", weights_only=False)
cfg = DataFreeConfig.from_dict(state["config"])
model = ModulatedSphereDecoder(cfg)
model.load_state_dict(state["model"])
model.eval()
for p in model.parameters():
    p.requires_grad_(False)
target_trace = float(cfg.cov_trace_target)
gen = torch.Generator().manual_seed(11)


def normalised(field):
    pts = torch.rand(600000, 3, generator=gen) * 3 - 1.5
    inside = pts[field(pts) > 0]
    c = inside.mean(0)
    cov = torch.cov((inside - c).T)
    evals, V = torch.linalg.eigh(cov)
    s = float(torch.sqrt(evals.sum() / target_trace))
    return (lambda x: field((x @ V.T) * s + c) / s), s


def ball(n, r=0.9):
    x = torch.randn(n, 3, generator=gen)
    return x / x.norm(dim=-1, keepdim=True) * r * torch.rand(n, 1, generator=gen) ** (1 / 3)


plate = A.box([0, 0, 0], [0.6, 0.6, 0.02])
boards = [A.box([0, 0, zc], [0.5, 0.3, 0.02]) for zc in (-0.55, -0.18, 0.18, 0.55)]
sides = [A.box([-0.5, 0, 0], [0.02, 0.3, 0.6]), A.box([0.5, 0, 0], [0.02, 0.3, 0.6])]
sets = {
    "solids": {
        "cube": A.box([0, 0, 0], [0.4, 0.4, 0.4]),
        "box 5:4:3": A.box([0, 0, 0], [0.5, 0.4, 0.3]),
        "slab 3:3:1": A.box([0, 0, 0], [0.6, 0.6, 0.2]),
        "rod 4:1:1": A.capsule([-0.6, 0, 0], [0.6, 0, 0], 0.2),
        "L-block": A.union(A.box([0, 0, 0], [0.6, 0.2, 0.2]), A.box([-0.4, 0.4, 0], [0.2, 0.6, 0.2])),
        "dumbbell": A.union(A.sphere([-0.45, 0, 0], 0.3), A.sphere([0.45, 0, 0], 0.3), A.capsule([-0.45, 0, 0], [0.45, 0, 0], 0.15)),
    },
    "bar": {
        "thin sheet": plate,
        "plate with hole": A.intersection(plate, A.complement(A.sphere([0, 0, 0], 0.25))),
        "open shelf": A.union(*boards, *sides),
        "shelf with back": A.union(*boards, *sides, A.box([0, 0.28, 0], [0.5, 0.02, 0.6])),
        "ring": A.torus(0.55, 0.2),
        "bent sheet": A.union(A.box([0, 0, 0], [0.6, 0.5, 0.02]), A.box([0.58, 0, 0.3], [0.02, 0.5, 0.3])),
    },
}
starts, n_fit = 6, 700
x_eval = ball(40000)
print("run", run, ckpt, "step", state["step"], "| set", which, "| IoU over the ball |x|<0.9")
print(f"{'target':16s} {'IoU fit':>8s} {'IoU sphere':>10s} {'false free':>10s} {'false occ.':>10s} {'centre ball solid?':>18s} {'t(s)':>6s}")
ious = []
for name, field in sets[which].items():
    t0 = time.time()
    tgt, s = normalised(field)
    z = torch.nn.Parameter(F.normalize(torch.randn(starts, cfg.latent_dim, generator=gen), dim=-1))
    opt = torch.optim.Adam([z], lr=0.05)
    for it in range(steps):
        pool = ball(4000)
        with torch.no_grad():
            yp = tgt(pool)
            near = yp.abs().argsort()[: n_fit // 2]
            rest = torch.randperm(4000, generator=gen)[: n_fit - n_fit // 2]
            idx = torch.cat([near, rest])
            x = pool[idx]; y = yp[idx].clamp(-0.1, 0.1)
        pred = model(z, x[None].expand(starts, -1, -1)).clamp(-0.1, 0.1)
        loss = (pred - y[None]).abs().mean(dim=1)
        opt.zero_grad()
        loss.sum().backward()
        opt.step()
    with torch.no_grad():
        ye = tgt(x_eval) > 0
        best_iou, best = -1.0, None
        for k in range(starts):
            pe = model(z[k:k + 1].detach(), x_eval[None])[0] > 0
            iou = float((ye & pe).sum() / (ye | pe).sum())
            if iou > best_iou:
                best_iou, best = iou, pe
        sph = x_eval.norm(dim=-1) < 0.5
        iou_s = float((ye & sph).sum() / (ye | sph).sum())
        ff = float((ye & ~best).sum() / ye.sum())
        fo = float((~ye & best).sum() / ye.sum())
        inner = x_eval[x_eval.norm(dim=-1) < (cfg.base_radius - cfg.correction_bound)]
        ok = bool((tgt(inner) > 0).all())
    ious.append(best_iou)
    print(f"{name:16s} {best_iou:8.3f} {iou_s:10.3f} {ff:10.3f} {fo:10.3f} {str(ok):>18s} {time.time()-t0:6.1f}")
print(f"mean IoU {np.mean(ious):.3f}   (agreed bar: mean novel-target IoU >= 0.70)")
