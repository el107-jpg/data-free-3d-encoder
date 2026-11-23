"""CPU-only. Fit codes of a trained data-free decoder to a few pose-normalised analytic solids.
Not a benchmark: a preview of the 'fit held-out real shapes' evaluation, to see the order of magnitude."""
import sys, time
import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(6)
ROOT = "/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/"
run = sys.argv[1]
from arm_planner.data_free.config import DataFreeConfig
from arm_planner.data_free.decoder import ModulatedSphereDecoder
from arm_planner.data_free import analytic_shapes as A

state = torch.load(f"{ROOT}{run}/checkpoint_last.pt", map_location="cpu", weights_only=False)
cfg = DataFreeConfig.from_dict(state["config"])
model = ModulatedSphereDecoder(cfg)
model.load_state_dict(state["model"])
model.eval()
for p in model.parameters():
    p.requires_grad_(False)
target_trace = float(cfg.cov_trace_target)
gen = torch.Generator().manual_seed(11)


def normalised(field):
    """Same pose convention as training: centroid 0, covariance trace = target, principal axes on x, y, z."""
    pts = torch.rand(400000, 3, generator=gen) * 3 - 1.5
    inside = pts[field(pts) > 0]
    c = inside.mean(0)
    cov = torch.cov((inside - c).T)
    evals, V = torch.linalg.eigh(cov)
    s = float(torch.sqrt(evals.sum() / target_trace))
    return (lambda x: field((x @ V.T) * s + c) / s), s


raw = {
    "cube": A.box([0, 0, 0], [0.4, 0.4, 0.4]),
    "box 5:4:3": A.box([0, 0, 0], [0.5, 0.4, 0.3]),
    "slab 3:3:1": A.box([0, 0, 0], [0.6, 0.6, 0.2]),
    "rod 4:1:1": A.capsule([-0.6, 0, 0], [0.6, 0, 0], 0.2),
    "L-block": A.union(A.box([0, 0, 0], [0.6, 0.2, 0.2]), A.box([-0.4, 0.4, 0], [0.2, 0.6, 0.2])),
    "dumbbell": A.union(A.sphere([-0.45, 0, 0], 0.3), A.sphere([0.45, 0, 0], 0.3), A.capsule([-0.45, 0, 0], [0.45, 0, 0], 0.15)),
}
starts, n_fit, steps = 6, 700, int(sys.argv[2]) if len(sys.argv) > 2 else 90
x_eval = torch.randn(20000, 3, generator=gen)
x_eval = x_eval / x_eval.norm(dim=-1, keepdim=True) * 0.9 * torch.rand(20000, 1, generator=gen) ** (1 / 3)
print("run", run, "| IoU and error rates over the ball |x|<0.9; 'false free' = share of the solid's volume the fit calls empty")
print(f"{'shape':12s} {'IoU fit':>8s} {'IoU sphere':>10s} {'false free':>10s} {'false occ.':>10s} {'origin ball inside?':>20s} {'t(s)':>6s}")
for name, field in raw.items():
    t0 = time.time()
    tgt, s = normalised(field)
    z = torch.nn.Parameter(F.normalize(torch.randn(starts, cfg.latent_dim, generator=gen), dim=-1))
    opt = torch.optim.Adam([z], lr=0.05)
    for it in range(steps):
        x = torch.randn(n_fit, 3, generator=gen)
        x = x / x.norm(dim=-1, keepdim=True) * 0.9 * torch.rand(n_fit, 1, generator=gen) ** (1 / 3)
        with torch.no_grad():
            y = tgt(x).clamp(-0.1, 0.1)
        pred = model(z, x[None].expand(starts, -1, -1)).clamp(-0.1, 0.1)
        loss = (pred - y[None]).abs().mean(dim=1)
        opt.zero_grad()
        loss.sum().backward()
        opt.step()
    best = int(loss.argmin())
    with torch.no_grad():
        ye = tgt(x_eval) > 0
        pe = model(z[best:best + 1].detach(), x_eval[None])[0] > 0
        sph = x_eval.norm(dim=-1) < 0.5
        iou = float((ye & pe).sum() / (ye | pe).sum())
        iou_s = float((ye & sph).sum() / (ye | sph).sum())
        ff = float((ye & ~pe).sum() / ye.sum())
        fo = float((~ye & pe).sum() / ye.sum())
        ball = x_eval[x_eval.norm(dim=-1) < (cfg.base_radius - cfg.correction_bound)]
        ok = bool((tgt(ball) > 0).all())
    print(f"{name:12s} {iou:8.3f} {iou_s:10.3f} {ff:10.3f} {fo:10.3f} {str(ok):>20s} {time.time()-t0:6.1f}   final L1 per start: {np.array2string(loss.detach().numpy(), precision=4)}")
