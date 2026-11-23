"""What kind of variation does the latent produce, and is rotation actually pinned?

For each checkpoint:
 A. cross-code statistics on a grid (64 codes): centroid, size (covariance trace), second-moment
    anisotropy, off-diagonal correlations.
 B. per-code decomposition of surface motion per latent step (8 probe codes x 31 tangent
    directions) into translation / rotation / uniform scale / remaining (real deformation).
"""
import json
import sys

import torch

from arm_planner.data_free.field import field_and_grad
from arm_planner.data_free.latent import tangent_basis
from arm_planner.data_free.train import Trainer

out = {}
for label, ckpt in zip(sys.argv[1::2], sys.argv[2::2]):
    trainer = Trainer.from_checkpoint(ckpt, device="cuda", tensorboard=False)
    cfg, model = trainer.cfg, trainer.model.eval()
    codes = trainer.bank.codes

    # ---------- A. cross-code statistics
    res = 56
    ax = (torch.arange(res, device="cuda") + 0.5) / res * 2 - 1
    grid = torch.stack(torch.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3)
    cents, traces, eigs, corrs = [], [], [], []
    with torch.no_grad():
        for c0 in range(0, 64, 4):
            f = model(codes[c0:c0 + 4], grid[None].expand(4, -1, -1))
            w = (f > 0).float()
            tot = w.sum(1, keepdim=True)
            cen = (w[..., None] * grid).sum(1) / tot
            d = grid[None] - cen[:, None]
            cov = torch.einsum("bn,bni,bnj->bij", w, d, d) / tot[..., None]
            cents.append(cen); traces.append(torch.diagonal(cov, dim1=1, dim2=2).sum(1))
            eigs.append(torch.linalg.eigvalsh(cov))
            sd = torch.diagonal(cov, dim1=1, dim2=2).sqrt()
            corr = cov / (sd[:, :, None] * sd[:, None, :])
            corrs.append(torch.stack([corr[:, 0, 1], corr[:, 0, 2], corr[:, 1, 2]], 1))
    cen, tr, eig, corr = torch.cat(cents), torch.cat(traces), torch.cat(eigs), torch.cat(corrs)
    aniso = (eig[:, 2] - eig[:, 0]) / eig.mean(1)                 # 0 = perfectly round second moments
    stats = {
        "centroid_offset_mean": float(cen.norm(dim=1).mean()), "centroid_offset_max": float(cen.norm(dim=1).max()),
        "size_trace_mean": float(tr.mean()), "size_trace_rel_std": float(tr.std() / tr.mean()),
        "size_trace_rel_range": [float(tr.min() / tr.mean()), float(tr.max() / tr.mean())],
        "elongation_mean": float((eig[:, 2] / eig[:, 1]).sqrt().mean()),
        "flatness_mean": float((eig[:, 1] / eig[:, 0]).sqrt().mean()),
        "anisotropy_mean": float(aniso.mean()), "anisotropy_max": float(aniso.max()),
        "offdiag_corr_abs_mean": float(corr.abs().mean()), "offdiag_corr_abs_max": float(corr.abs().max()),
    }

    # ---------- B. decomposition of surface motion per latent step
    gen = torch.Generator().manual_seed(7)
    shares = []
    step = 1e-3
    for k in range(8):
        z = codes[k]
        x = (torch.rand(1, 6000, 3, generator=gen) * 2 - 1).cuda()
        mod = model.prepare(z[None])
        f, g = field_and_grad(model, mod, x, create_graph=False)
        xp = (x - f[..., None] * g).detach()
        fp, gp = field_and_grad(model, mod, xp, create_graph=False)
        keep = (fp[0].abs() < 3 * fp[0].abs().median().clamp_min(1e-4)) & (gp[0].norm(dim=-1) > 0.3)
        p, gn = xp[0][keep], gp[0][keep]
        n_out = -gn / gn.norm(dim=-1, keepdim=True)
        T = tangent_basis(z)                                             # 31 x 32
        zs = torch.cat([z + step * T, z - step * T])
        zs = zs / zs.norm(dim=-1, keepdim=True)
        with torch.no_grad():
            vals = torch.cat([model(zs[i:i + 8], p[None].expand(zs[i:i + 8].shape[0], -1, -1))
                              for i in range(0, zs.shape[0], 8)])
        df = (vals[:31] - vals[31:]) / (2 * step)                        # 31 x P
        D = (df / gn.norm(dim=-1)[None]).T                               # P x 31 outward displacement per unit code step
        eye = torch.eye(3, device="cuda")
        Bt = n_out @ eye                                                  # translations: n . e_i
        Br = torch.stack([(n_out * torch.linalg.cross(eye[i].expand_as(p), p)).sum(-1) for i in range(3)], 1)
        Bs = (n_out * p).sum(-1, keepdim=True)
        B = torch.cat([Bt, Br, Bs], 1)                                    # P x 7
        C = torch.linalg.lstsq(B, D).solution                            # 7 x 31
        total = D.square().sum()
        part = lambda cols: float((B[:, cols] @ C[cols]).square().sum() / total)
        resid = float((D - B @ C).square().sum() / total)
        # how well-conditioned is the rotation fit: a perfectly round surface has no rotation signal
        rot_signal = float(Br.square().mean().sqrt())                    # rms of n . (e x p); 0 on a sphere
        shares.append({"translation": part([0, 1, 2]), "rotation": part([3, 4, 5]), "scale": part([6]),
                       "deformation": resid, "rms_motion": float(D.square().mean().sqrt()),
                       "rotation_signal": rot_signal, "points": int(p.shape[0])})
    mean = {k: sum(s[k] for s in shares) / len(shares) for k in shares[0]}
    out[label] = {"cross_code": stats, "motion_shares_mean": mean}
    print("==", label)
    print("  cross-code:", {k: (round(v, 4) if isinstance(v, float) else [round(x, 3) for x in v]) for k, v in stats.items()})
    print("  motion per latent step:", {k: round(v, 4) for k, v in mean.items()})
    del trainer
    torch.cuda.empty_cache()
json.dump(out, open("/home/eric-luo/.claude/jobs/8dde3c62/tmp/diversity_kinds.json", "w"), indent=1)
