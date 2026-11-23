"""Effective rank of the main network, per hidden layer.  usage: main_rank.py <label>=<checkpoint> [...]
64 codes x 4,096 points (radius 0.2-0.8). Per layer (after the activation):
  all     - effective rank (exp entropy of normalised squared singular values) of the 262,144 x 512 activations,
            plus the number of directions for 90% / 99% of the variance;
  by code - the same for the code-dependent part (activation minus that point's mean over the 64 codes);
  weights - effective rank of the layer's weight matrix (its square part, 512 x 512)."""
import sys, torch
from arm_planner.data_free.train import Trainer
def erank_cov(C):
    s = torch.linalg.eigvalsh(C.double()).clamp_min(0).flip(0); p = s / s.sum(); c = torch.cumsum(p, 0)
    return float(torch.exp(-(p * torch.log(p.clamp_min(1e-30))).sum())), int((c < 0.9).sum()) + 1, int((c < 0.99).sum()) + 1
def erank_w(W):
    s = torch.linalg.svdvals(W.double()); p = s.square() / s.square().sum()
    return float(torch.exp(-(p * torch.log(p.clamp_min(1e-30))).sum()))
for item in sys.argv[1:]:
    label, path = item.split("=", 1)
    t = Trainer.from_checkpoint(path, device="cuda", tensorboard=False)
    m, cfg = t.model.eval(), t.cfg
    codes = torch.nn.functional.normalize(torch.randn(64, cfg.latent_dim, generator=torch.Generator().manual_seed(3)), dim=-1).cuda()
    gen = torch.Generator().manual_seed(4)
    x = torch.randn(4096, 3, generator=gen); x = x / x.norm(dim=-1, keepdim=True) * (0.2 + 0.6 * torch.rand(4096, 1, generator=gen)); x = x.cuda()
    acts = {}
    with torch.no_grad():
        for s in range(0, 64, 8):
            got = []
            m._net(m.prepare(codes[s:s + 8]), x[None].expand(8, -1, -1), collect=got)
            for i, z in enumerate(got): acts.setdefault(i, []).append(m.activation(z).detach().cpu())
    print(f"\n{label}: layer | all: eff / 90% / 99% | by code: eff / 90% / 99% | code share of variance | weights eff")
    for i in range(len(acts)):
        a = torch.cat(acts[i], 0).double()                          # 64 x 4096 x W
        W = a.shape[-1]
        flat = a.reshape(-1, W); flat = flat - flat.mean(0)
        C_all = flat.T @ flat / flat.shape[0]
        d = a - a.mean(0, keepdim=True); d = d.reshape(-1, W)
        C_code = d.T @ d / d.shape[0]
        w = m.hidden[i].weight.detach()
        w = w[:, :W] if w.shape[1] >= W else w
        e_all, e_code = erank_cov(C_all), erank_cov(C_code)
        share = float(torch.trace(C_code) / torch.trace(C_all))
        print(f"   {i}: {e_all[0]:6.1f} / {e_all[1]:3d} / {e_all[2]:3d} | {e_code[0]:6.1f} / {e_code[1]:3d} / {e_code[2]:3d} | {share:.2f} | {erank_w(w):6.1f}")
