import sys, time
src = open(__file__.replace("pocket_check.py", "blind.py")).read().split("out = []")[0]
src = src.replace('__file__.replace("blind.py", "compare.py")', repr(__file__.replace("pocket_check.py", "compare.py")))
exec(src)
ins, outs, times, probe_ms = [], [], [], []
for k in range(N_SHAPES):
    fn = make_field(k); LOG.clear()
    gen = torch.Generator(device=dev).manual_seed(1000 + k)
    p, _, ok = land(fn, shell_points(gen, 192)); landed = p[ok].cpu().numpy()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    lab, ids, st = pockets(landed)
    torch.cuda.synchronize(); times.append((time.perf_counter() - t0) * 1000)
    m = np.isin(lab, ids).reshape(-1)
    f = f_only(fn, CELLS).cpu().numpy()
    ins.append(float((f[m] > 0).mean()) if m.any() else 0.); outs.append(int(m.sum()))
print(f"{kind}: pocket cells inside the solid {np.mean(ins):.2f} (per shape 10-90%: {np.quantile(ins,.1):.2f}-{np.quantile(ins,.9):.2f}); pocket cells per shape {np.mean(outs):.0f}; pocket search {np.mean(times):.0f} ms per shape per round (CPU numpy)")
