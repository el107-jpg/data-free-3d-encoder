"""How much do shell samples (counted equally) bias the pose/volume estimates, on J4 shapes at 5,000?"""
import copy, torch
from arm_planner.data_free.train import Trainer
from arm_planner.data_free.field import sample_points
from arm_planner.data_free import losses as LS
t = Trainer.from_checkpoint("/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/J4_d64_nocentre_procedural96_even_code_leak05_beta100_fourier3_5k/checkpoint_last.pt", device="cuda", tensorboard=False)
model, cfg = t.model.eval(), t.cfg
codes = torch.nn.functional.normalize(torch.randn(64, cfg.latent_dim, generator=torch.Generator().manual_seed(2024)), dim=-1).cuda()
class FB: pass
from arm_planner.data_free.field import shell_volume_weight
def moments_with(mode, n=65536, weighted=False):
    c = copy.copy(cfg); c.point_sampling = mode; c.points = n
    x = sample_points(c, 1, torch.Generator(device="cuda").manual_seed(5), "cuda")[0, 1:]
    out = []
    with torch.no_grad():
        for s in range(0, 64, 1):
            f = torch.cat([model(codes[s:s + 1], x[None, j:j + 16384]) for j in range(0, len(x), 16384)], 1)
            fb = FB(); fb.f = f; fb.x = x[None]
            w = LS.soft_inside(fb, cfg)
            if weighted: w = w * shell_volume_weight(fb.x, c)
            tot = w.sum(1, keepdim=True)
            cen = (w[..., None] * fb.x).sum(1) / tot
            off = fb.x - cen[:, None]
            cov = torch.einsum("bn,bni,bnj->bij", w, off, off) / tot[..., None]
            frac = w.mean(1) * (8.0 if mode == "uniform" else 1.0)          # fraction of the box vs of its sample
            out.append(torch.cat([cen.norm(dim=-1, keepdim=True), torch.diagonal(cov, dim1=1, dim2=2).sum(-1, keepdim=True),
                                  (cov - torch.diag_embed(torch.diagonal(cov, dim1=1, dim2=2))).square().sum((1, 2))[:, None], w.mean(1, keepdim=True)], 1))
    return torch.cat(out)
u, s, sw = moments_with("uniform"), moments_with("shells"), moments_with("shells", weighted=True)
for i, name in enumerate(("centroid offset |c|", "size: covariance trace", "axes: sum sq off-diagonal", "inside fraction (box)")):
    a, b, c_ = u[:, i], s[:, i], sw[:, i]
    if i == 3: a = a * 8 / 8; b = b; c_ = c_
    print(f"{name:24s} uniform {a.mean():.4f}  shells {b.mean():.4f} (x{(b / a.clamp_min(1e-9)).mean():.2f})  shells weighted {c_.mean():.4f} (x{(c_ / a.clamp_min(1e-9)).mean():.2f})")
print("J4 targets: covariance trace", cfg.cov_trace_target, "; centroid tolerance 2.5e-4 on |c|^2 (|c| ~ 0.016)")
