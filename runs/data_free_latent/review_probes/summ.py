import json, sys, statistics as st

ROOT = "/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/"
run = sys.argv[1]
rows = [json.loads(l) for l in open(f"{ROOT}{run}/metrics.jsonl")]
n = len(rows)
keys = [k for k in rows[0] if k.startswith("term/")]
names = [k[5:] for k in keys]
edges = [0, 50, 100, 250, 500, 1000, 2000, 3000, 4000, 5000]
edges = [e for e in edges if e < n] + [n]


def mean(xs):
    xs = [x for x in xs if x == x]
    return sum(xs) / max(len(xs), 1)


print("run", run, "rows", n)
hdr = "win        lr       gnorm   loss     div     " + " ".join(f"{nm[:9]:>18}" for nm in names)
print(hdr)
for a, b in zip(edges[:-1], edges[1:]):
    w = rows[a:b]
    s = f"{a:5d}-{b:5d} {mean([r['lr'] for r in w]):.1e} {mean([r['grad_norm'] for r in w]):7.2f} {mean([r['loss'] for r in w]):8.3f} {mean([r['diversity'] for r in w]):7.3f} "
    for nm in names:
        t = mean([r['term/' + nm] for r in w]); m = mean([r['mult/' + nm] for r in w])
        s += f" {t:8.1e}x{m:7.1f}".rjust(18)
    print(s)
w = rows[-500:]
print("last-500 weighted contributions (mult*term):")
for nm in names:
    c = mean([r['term/' + nm] * r['mult/' + nm] for r in w])
    print(f"   {nm:14s} {c:9.4f}  term {mean([r['term/'+nm] for r in w]):.2e}  mult {mean([r['mult/'+nm] for r in w]):8.1f}")
print("   diversity", mean([r['diversity'] for r in w]), "spread", mean([r.get('spread', 0) for r in w]), "loss", mean([r['loss'] for r in w]))
gn = [r['grad_norm'] for r in rows if r['grad_norm'] == r['grad_norm']]
print("frac steps grad_norm>1 (clipped):", sum(g > 1 for g in gn) / n, "median gnorm", st.median(gn), "p10", sorted(gn)[len(gn)//10], "p90", sorted(gn)[9*len(gn)//10])
print("skipped", sum(r['skipped'] for r in rows))
for k in ("div/mean_ratio", "div/dead_zone_fraction", "div/mean_shape_distance", "div/mean_latent_distance", "div/nearest_distance", "div/order_violations", "e", "spread", "spread_corr", "grad_change"):
    if k in rows[0]:
        print(f"  {k}: " + " ".join(f"{mean([r.get(k,0) for r in rows[a:b]]):.4f}" for a, b in zip(edges[:-1], edges[1:])))
gk = [k for k in rows[0] if k.startswith("graph/")]
for k in gk:
    print(f"  {k}: " + " ".join(f"{mean([r.get(k,0) for r in rows[a:b]]):.2f}" for a, b in zip(edges[:-1], edges[1:])))
print("graph_mode counts", {m: sum(r['graph_mode'] == m for r in rows) for m in ('off', 'sign', 'full')})
tk = [k for k in rows[0] if k.startswith("time_ms/")]
print("time", {k: round(mean([r[k] for r in rows[-500:]]), 1) for k in tk}, "gpu MB", mean([r['gpu_peak_mb'] for r in rows[-500:]]))
