"""Where is the decoder's capacity limited?  usage: capacity.py <label>=<checkpoint> [...]
1. Code path: effective rank (exp of the entropy of the normalised singular values, and the count for 90/99% of the
   variance) of the mapping output (256) and of the FiLM scales and shifts (8 x 512 x 2) over 4,096 random codes.
2. Main network: per hidden layer, over 64 codes x 4,096 points near the surfaces: share of units that are essentially
   never active (pre-activation > 0 on < 1% of samples), and share whose response barely changes between codes
   (between-code variance < 5% of total variance)."""
import sys, torch
from arm_planner.data_free.train import Trainer
def erank(X):
    X = X - X.mean(0, keepdim=True)
    s = torch.linalg.svdvals(X.double()); p = s.square() / s.square().sum()
    c = torch.cumsum(p, 0)
    return float(torch.exp(-(p * torch.log(p.clamp_min(1e-30))).sum())), int((c < 0.9).sum()) + 1, int((c < 0.99).sum()) + 1
for item in sys.argv[1:]:
    label, path = item.split("=", 1)
    t = Trainer.from_checkpoint(path, device="cuda", tensorboard=False)
    m, cfg = t.model.eval(), t.cfg
    codes = torch.nn.functional.normalize(torch.randn(4096, cfg.latent_dim, generator=torch.Generator().manual_seed(3)), dim=-1).cuda()
    with torch.no_grad():
        h = codes
        for layer in m.mapping: h = layer(h)
        film = m.film(h)
    print(f"\n{label}: effective rank (entropy / 90% / 99%): mapping output (256 wide) {erank(h)}, FiLM vectors (8192) {erank(film)}")
    L = cfg.layers; per = film.shape[1] // L
    print("   FiLM per layer:", [erank(film[:, i * per:(i + 1) * per])[0].__round__(1) for i in range(L)])
    # main network unit usage via forward hooks on hidden pre-activations
    acts = {}
    sub = codes[:64]
    gen = torch.Generator().manual_seed(4)
    x = torch.randn(4096, 3, generator=gen); x = x / x.norm(dim=-1, keepdim=True) * (0.2 + 0.6 * torch.rand(4096, 1, generator=gen))
    x = x.cuda()
    with torch.no_grad():
        for s in range(0, 64, 8):
            got = []
            m._net(m.prepare(sub[s:s + 8]), x[None].expand(8, -1, -1), collect=got)
            for i, z in enumerate(got): acts.setdefault(i, []).append(z.detach())
    out = []
    for i in range(L):
        a = torch.cat([z.reshape(8, 4096, -1) for z in acts[i]], 0)          # codes x points x units (pre-activation incl. FiLM?)
        on = (a > 0).float().mean((0, 1))
        total = a.var(dim=(0, 1)); between = a.mean(1).var(0)
        out.append((float((on < 0.01).float().mean()), float(((between / total.clamp_min(1e-12)) < 0.05).float().mean()), float((between / total.clamp_min(1e-12)).mean())))
    print("   hidden layers: never-active share / code-insensitive share / mean between-code variance share")
    for i, (d, c, b) in enumerate(out):
        print(f"     layer {i}: {d:.2f} / {c:.2f} / {b:.2f}")
