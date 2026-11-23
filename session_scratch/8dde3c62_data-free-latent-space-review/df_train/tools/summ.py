import json, sys, statistics as st
p = sys.argv[1]
rows = [json.loads(l) for l in open(p + "/metrics.jsonl")]
def med(k, sel=lambda r: True):
    v = [r[k] for r in rows if sel(r)]
    return st.median(v) if v else float('nan')
skip = 10
late = lambda r: r["step"] >= skip
print("steps", len(rows))
for k in ["total", "field", "graph", "confirm", "losses", "backward"]:
    print(f"{k:9s} all {med('time_ms/'+k, late):7.1f}  build-steps {med('time_ms/'+k, lambda r: late(r) and r['graph_built']):7.1f}  reuse-steps {med('time_ms/'+k, lambda r: late(r) and not r['graph_built']):7.1f}")
print("graph/build_ms (internal, unsynced) median", med("graph/build_ms", lambda r: late(r) and r['graph_built']))
print("peak MB max", max(r["gpu_peak_mb"] for r in rows), "median", med("gpu_peak_mb"))
print("modes", {m: sum(r["graph_mode"] == m for r in rows) for m in ("off", "sign", "full")}, "skipped", sum(r["skipped"] for r in rows))
if len(sys.argv) > 2:
    pts = [int(a) for a in sys.argv[2:]]
    w = 20
    keys = ["diversity", "div/mean_ratio", "div/dead_zone_fraction", "term/eikonal", "term/consistency", "e", "term/barrier", "term/pose", "term/origin", "term/volume", "grad_change", "grad_norm", "loss"] + \
        ["term/" + k for k in ("bridge", "delete", "fill", "widen", "thicken")] + \
        ["graph/" + k for k in ("fragments_kind0", "fragments_kind1", "cavities", "gap_sites", "wall_sites", "confirm_dropped_kind0", "confirm_dropped_cavities")] + \
        ["mult/" + k for k in ("eikonal", "consistency", "barrier", "origin", "volume", "pose", "bridge", "delete", "fill", "widen", "thicken")]
    print("window mean over", w, "steps starting at", pts)
    for k in keys:
        out = []
        for s in pts:
            seg = [r[k] for r in rows if s <= r["step"] < s + w]
            out.append(sum(seg) / len(seg) if seg else float('nan'))
        print(f"{k:32s}", "  ".join(f"{v:10.4g}" for v in out))
    print("mode at pts", [[r["graph_mode"] for r in rows if r["step"] == min(s, len(rows)-1)] for s in pts])
    tot = {k: sum(r["graph/" + k] for r in rows if r["graph_built"]) for k in ("fragments_kind0", "fragments_kind1", "cavities", "gap_sites", "wall_sites", "confirm_dropped_kind0", "confirm_dropped_kind1", "confirm_dropped_cavities")}
    print("totals over builds", tot)
    try:
        for l in open(p + "/diagnostics.jsonl"):
            d = json.loads(l)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items() if not k.startswith("descriptor_mean") and not k.startswith("descriptor_std")})
    except FileNotFoundError:
        pass
