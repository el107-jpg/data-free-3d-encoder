import sys
here = __file__
src = open(here.replace("shell_weights.py", "batch_shared.py")).read().split("cfg = trainer.cfg\nout = []")[0]
src = src.replace('here.replace("batch_shared.py", "blind.py")', repr(here.replace("shell_weights.py", "blind.py")))
src = src.replace('here.replace("batch_shared.py", "compare.py")', repr(here.replace("shell_weights.py", "compare.py")))
exec(src)
from arm_planner.data_free.field import sample_points
lim = (0, .2, .55, .7, .9); frac = (.05, .5, .2, .25)
def band_weight(x):                                    # SD2's per-point weight: band volume share / band point share
    r = x.norm(dim=-1).cpu().numpy(); w = np.zeros(len(r))
    for lo, hi, fr in zip(lim, lim[1:], frac):
        m = (r >= lo) & (r < hi); w[m] = ((hi**3 - lo**3) / .9**3) / fr
    return w / w.sum()
eq, wt, un = [], [], []
for k in range(N_SHAPES):
    fn = make_field(k); true, w, deep = truth(fn)
    xs = shell_points(torch.Generator(device=dev).manual_seed(950 + k), 2048)
    _, idx = cKDTree(true).query(xs.cpu().numpy())
    bw = band_weight(xs)
    eq.append(deep[idx].mean()); wt.append((bw * deep[idx]).sum())
    u = sample_points(trainer.cfg, 1, torch.Generator(device=dev).manual_seed(970 + k), dev)[0]
    u = u[u.norm(dim=-1) < .9]; _, iu = cKDTree(true).query(u.cpu().numpy()); un.append(deep[iu].mean())
print(f"share of the distance's weight on points that see the most recessed quarter (25% of the area), {len(eq)} J4 shapes:")
print(f"  2048 uniform (current J4, |x| < 0.9):        {np.mean(un):.3f}")
print(f"  2048 shells, volume-weighted (as SD2):       {np.mean(wt):.3f}")
print(f"  2048 shells, counted equally:                {np.mean(eq):.3f}")
