"""Are the used latent directions the same fixed directions of code space at every code?

For each probe code: surface motion per unit chord along the 31 tangent directions (pose removed),
expressed in the 32 ambient code coordinates -> a 32x32 matrix G_c of motion per direction.
A global set of directions is fitted on half of the codes and tested on the other half:
how much of each held-out code's motion lies inside the global top-k directions.
"""
import sys

import numpy as np
import torch

from arm_planner.data_free.field import field_and_grad
from arm_planner.data_free.latent import tangent_basis
from arm_planner.data_free.train import Trainer

ckpt = sys.argv[1]
n_codes = int(sys.argv[2]) if len(sys.argv) > 2 else 32
trainer = Trainer.from_checkpoint(ckpt, device="cuda", tensorboard=False)
model = trainer.model.eval()
gen = torch.Generator().manual_seed(11)
codes = torch.nn.functional.normalize(torch.randn(n_codes, 32, generator=gen), dim=-1).cuda()
step = 1e-3
grams, tops = [], []
for k in range(n_codes):
    z = codes[k]
    x = (torch.rand(1, 4000, 3, generator=gen) * 2 - 1).cuda()
    mod = model.prepare(z[None])
    f, g = field_and_grad(model, mod, x, create_graph=False)
    xp = (x - f[..., None] * g).detach()
    fp, gp = field_and_grad(model, mod, xp, create_graph=False)
    keep = (fp[0].abs() < 3 * fp[0].abs().median().clamp_min(1e-4)) & (gp[0].norm(dim=-1) > 0.3)
    p, gn = xp[0][keep], gp[0][keep]
    n_out = -gn / gn.norm(dim=-1, keepdim=True)
    T = tangent_basis(z)
    zs = torch.cat([z + step * T, z - step * T])
    zs = zs / zs.norm(dim=-1, keepdim=True)
    with torch.no_grad():
        vals = torch.cat([model(zs[i:i + 8], p[None].expand(zs[i:i + 8].shape[0], -1, -1))
                          for i in range(0, zs.shape[0], 8)])
    D = ((vals[:31] - vals[31:]) / (2 * step) / gn.norm(dim=-1)[None]).T
    eye = torch.eye(3, device="cuda")
    B = torch.cat([n_out @ eye,
                   torch.stack([(n_out * torch.linalg.cross(eye[i].expand_as(p), p)).sum(-1) for i in range(3)], 1),
                   (n_out * p).sum(-1, keepdim=True)], 1)
    R = D - B @ torch.linalg.lstsq(B, D).solution
    J = (R / np.sqrt(p.shape[0])).double() @ T.double()               # P x 32 in ambient code coordinates
    G = (J.T @ J).cpu().numpy()
    grams.append(G)
    w, v = np.linalg.eigh(G)
    tops.append((w[::-1], v[:, ::-1]))
grams = np.array(grams)
half = n_codes // 2


def top_dirs(gs, k):
    w, v = np.linalg.eigh(gs.mean(0))
    return v[:, ::-1][:, :k], w[::-1]


def capture(G, U):
    return float(np.trace(U.T @ G @ U) / np.trace(G))


print(f"update {trainer.step_count}; {n_codes} fresh random codes; global directions fitted on the first {half}, tested on the rest")
U_all, w_all = top_dirs(grams[:half], 32)
print("global motion per direction (fitted half), top 14:", " ".join(f"{x:.4f}" for x in w_all[:14]))
for k in (1, 3, 5, 7, 10, 12):
    U = U_all[:, :k]
    held = [capture(G, U) for G in grams[half:]]
    own = [float(w[:k].sum() / w.sum()) for w, _ in tops[half:]]
    rand = k / 31
    print(f"  top {k:2d} global directions hold {np.mean(held):.3f} of a held-out code's motion "
          f"(min {np.min(held):.3f}); its own best {k} hold {np.mean(own):.3f}; random directions would hold {rand:.3f}")
# does each code's single strongest direction coincide with the global strongest one?
u1 = U_all[:, 0]
cos1 = [abs(float(v[:, 0] @ u1)) for _, v in tops[half:]]
in7 = [float(np.linalg.norm(U_all[:, :7].T @ v[:, 0])) for _, v in tops[half:]]
in10 = [float(np.linalg.norm(U_all[:, :10].T @ v[:, 0])) for _, v in tops[half:]]
print(f"each code's strongest direction vs the global strongest: |cos| mean {np.mean(cos1):.2f} (range {np.min(cos1):.2f}..{np.max(cos1):.2f})")
print(f"each code's strongest direction lies inside the global top 7 by {np.mean(in7):.3f}, top 10 by {np.mean(in10):.3f} (1 = fully inside)")
# the unused set: how much motion do the weakest global directions get at any single code?
for k in (21, 15):
    U = U_all[:, -k:]
    held = [capture(G, U) for G in grams[half:]]
    print(f"  weakest {k} global directions hold {np.mean(held):.5f} of a held-out code's motion (max {np.max(held):.5f})")
# how much of each code lies in the used directions (the decoder only reads this part)
proj = (codes.cpu().numpy() @ U_all[:, :10])
print(f"length of a random code's part inside the top 10 directions: mean {np.linalg.norm(proj, axis=1).mean():.2f} "
      f"(range {np.linalg.norm(proj, axis=1).min():.2f}..{np.linalg.norm(proj, axis=1).max():.2f}); expected sqrt(10/32) = {np.sqrt(10/32):.2f}")
