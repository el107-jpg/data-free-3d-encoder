import math, torch
from arm_planner.data_free import losses as L
from arm_planner.data_free.config import DataFreeConfig
from arm_planner.data_free.field import FieldBatch
cfg = DataFreeConfig()
R = cfg.base_radius
print("hard 3R^2/5", 3*R*R/5, "soft_ball_trace", L.soft_ball_trace(R, cfg.soft_tau), "cfg target", cfg.cov_trace_target)
def fb_of(fn, n, seed, b=16):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(b, n, 3, generator=g) * 2 - 1
    x[:, 0] = 0
    return FieldBatch(codes=None, cluster=None, x=x, f=fn(x), g=None, xp=None, fp=None)
# large-N soft trace of the exact sphere
fb = fb_of(lambda x: R - x.norm(dim=-1), 2_000_000, 0, b=8)
_, c, cov = L.moments(fb, cfg)
tr = torch.diagonal(cov, dim1=-2, dim2=-1).sum(-1)
print("large-N soft trace of exact sphere: mean %.6f  se %.6f" % (tr.mean(), tr.std()/8**0.5))
th = 0.6
rot = torch.tensor([[math.cos(th), -math.sin(th), 0], [math.sin(th), math.cos(th), 0], [0, 0, 1.0]])
ax = torch.tensor([0.65, 0.5, 0.4])
def ellipsoid(x):  # not an exact SDF, fine for moments: scaled-space distance
    y = (x @ rot) / ax
    return (1 - y.norm(dim=-1)) * ax.min()
cases = {
    "centred sphere R": lambda x: R - x.norm(dim=-1),
    "shifted 0.05 in x": lambda x: R - (x - torch.tensor([0.05, 0, 0])).norm(dim=-1),
    "scaled 1.2 R": lambda x: 1.2 * R - x.norm(dim=-1),
    "rotated ellipsoid": ellipsoid,
}
for name, fn in cases.items():
    # truth from one large sample (soft moments, squared directly: noise negligible at this N)
    fbL = fb_of(fn, 2_000_000, 123, b=4)
    _, c, cov = L.moments(fbL, cfg)
    d = torch.diagonal(cov, dim1=-2, dim2=-1)
    truth = {"pose_centroid": float(c.square().sum(-1).mean()),
             "pose_scale": float((d.sum(-1) - cfg.cov_trace_target).square().mean()),
             "pose_axes": float((cov.square().sum((-2, -1)) - d.square().sum(-1)).mean())}
    vals = {k: [] for k in L.POSE_KEYS}; old = []
    for seed in range(400):
        fbs = fb_of(fn, 2048, seed, b=16)
        for k, v in L.pose_terms(fbs, cfg).items():
            vals[k].append(float(v))
        old.append(float(L.pose(fbs, cfg)))
    print(name)
    for k in L.POSE_KEYS:
        t = torch.tensor(vals[k])
        # each value is a mean over 16 independent shapes; per-shape std = std*4
        print(f"   {k:14s} truth {truth[k]: .3e}  mean {t.mean(): .3e}  se {t.std()/len(t)**0.5:.1e}  std(per shape) {t.std()*4:.2e}")
    print(f"   old pose (biased sum) mean {sum(old)/len(old):.3e}   truth sum {sum(truth.values()):.3e}")
