"""How noisy is the "total variance" figure? Per-code values over many random codes.

usage: variance_noise.py <checkpoint> [<checkpoint> ...]

evalrun.py reports the mean over 24 random codes of each code's total variance (sum over code directions of
the squared RMS surface motion per unit code step, pose removed). Here the same per-code number is computed
for 96 fresh codes: mean, spread between codes, standard error for 24 and for 96 codes, and what four separate
groups of 24 would each have reported. Also the per-code count of directions above 0.02.
"""
import sys

import numpy as np
import torch

from arm_planner.data_free.field import field_and_grad
from arm_planner.data_free.latent import tangent_basis
from arm_planner.data_free.train import Trainer

N, STEP, CHUNK = 96, 1e-3, 8
for path in sys.argv[1:]:
    trainer = Trainer.from_checkpoint(path, device="cuda", tensorboard=False)
    cfg, model = trainer.cfg, trainer.model.eval()
    dim = int(cfg.latent_dim)
    gen = torch.Generator().manual_seed(4242)
    codes = torch.nn.functional.normalize(torch.randn(N, dim, generator=gen), dim=-1).cuda()
    totals, solid, spectra = [], [], []
    for k in range(N):
        z = codes[k]
        x = (torch.rand(1, 5000, 3, generator=gen) * 2 - 1).cuda()
        mod = model.prepare(z[None])
        f, g = field_and_grad(model, mod, x, create_graph=False)
        xp = (x - f[..., None] * g).detach()
        fp, gp = field_and_grad(model, mod, xp, create_graph=False)
        keep = (fp[0].abs() < 3 * fp[0].abs().median().clamp_min(1e-4)) & (gp[0].norm(dim=-1) > 0.3)
        p, gn = xp[0][keep].detach(), gp[0][keep].detach()
        n_out = -gn / gn.norm(dim=-1, keepdim=True)
        T = tangent_basis(z)
        zs = torch.cat([z + STEP * T, z - STEP * T])
        zs = zs / zs.norm(dim=-1, keepdim=True)
        with torch.no_grad():
            vals = torch.cat([model(zs[i:i + CHUNK], p[None].expand(zs[i:i + CHUNK].shape[0], -1, -1))
                              for i in range(0, zs.shape[0], CHUNK)])
        half = dim - 1
        Dm = ((vals[:half] - vals[half:]) / (2 * STEP) / gn.norm(dim=-1)[None]).T
        eye = torch.eye(3, device="cuda")
        B = torch.cat([n_out @ eye,
                       torch.stack([(n_out * torch.linalg.cross(eye[i].expand_as(p), p)).sum(-1) for i in range(3)], 1),
                       (n_out * p).sum(-1, keepdim=True)], 1)
        R = Dm - B @ torch.linalg.lstsq(B, Dm).solution
        s2 = (torch.linalg.svdvals(R / np.sqrt(p.shape[0])) ** 2).cpu().numpy()
        totals.append(s2.sum()); solid.append(int((np.sqrt(s2) > 0.02).sum())); spectra.append(s2)
        del f, g, fp, gp
    t = np.array(totals)
    groups = t.reshape(4, 24).mean(1)
    pooled = np.sqrt(np.array(spectra).mean(0))
    print(f"{path.split('/')[-2][:34]} @ {trainer.step_count}: total variance over {N} codes: mean {t.mean():.3f}; between codes: std {t.std(ddof=1):.3f} "
          f"(range {t.min():.2f} to {t.max():.2f}); standard error with 24 codes {t.std(ddof=1) / np.sqrt(24):.3f}, with {N} codes {t.std(ddof=1) / np.sqrt(N):.3f}; "
          f"four groups of 24 would report {', '.join(f'{v:.3f}' for v in groups)}; solid directions from the pooled spectrum {int((pooled > 0.02).sum())}, "
          f"per code {np.mean(solid):.1f} (std {np.std(solid, ddof=1):.1f})", flush=True)
