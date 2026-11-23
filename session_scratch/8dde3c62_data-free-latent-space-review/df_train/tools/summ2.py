import json, sys, statistics as st
p = sys.argv[1]
rows = [json.loads(l) for l in open(p + "/metrics.jsonl")]
def med(k, sel=lambda r: True):
    v = [r[k] for r in rows if sel(r)]
    return st.median(v) if v else float('nan')
late = lambda r: r["step"] >= 10
print("steps", len(rows), " median ms total", round(med("time_ms/total", late), 1), " peak MB", round(max(r["gpu_peak_mb"] for r in rows)))
print("modes", {m: sum(r["graph_mode"] == m for r in rows) for m in ("off", "sign", "full")}, "skipped", sum(r["skipped"] for r in rows))
losses = [r["loss"] for r in rows]
print("loss min %.4g at step %d; steps with loss<0: %d; steps with loss<-0.01: %d" % (min(losses), losses.index(min(losses)), sum(l < 0 for l in losses), sum(l < -0.01 for l in losses)))
pk = ("pose_centroid", "pose_scale", "pose_axes")
for k in pk:
    v = [r["mult/" + k] * r["term/" + k] for r in rows]
    print(f"weighted {k}: min {min(v):.4g} max {max(v):.4g}; max mult {max(r['mult/'+k] for r in rows):.4g}")
pts = [int(a) for a in sys.argv[2:]]
w = 50
names = ("eikonal", "consistency", "barrier", "origin", "volume") + pk + ("bridge", "delete", "fill", "widen", "thicken")
keys = ["loss", "diversity", "div/mean_ratio", "div/dead_zone_fraction", "e", "grad_change", "grad_norm"] + ["term/" + k for k in names] + \
    ["graph/" + k for k in ("fragments_kind0", "fragments_kind1", "cavities", "gap_sites", "wall_sites", "extra_pieces", "confirm_dropped_kind0", "confirm_dropped_kind1", "confirm_dropped_cavities")] + \
    ["mult/" + k for k in names]
print("window mean over", w, "steps starting at", pts)
for k in keys:
    out = []
    for s in pts:
        seg = [r[k] for r in rows if s <= r["step"] < s + w]
        out.append(sum(seg) / len(seg) if seg else float('nan'))
    print(f"{k:32s}", "  ".join(f"{v:10.4g}" for v in out))
built = [r for r in rows if r["graph_built"]]
tot = {k: sum(r["graph/" + k] for r in built) for k in ("fragments_kind0", "fragments_kind1", "cavities", "gap_sites", "wall_sites", "confirm_dropped_kind0", "confirm_dropped_kind1", "confirm_dropped_cavities")}
print("totals over", len(built), "builds", tot)
for k in ("bridge", "delete", "fill", "widen", "thicken"):
    nz = [r for r in rows if r["term/" + k] > 0]
    if nz:
        v = [r["term/" + k] for r in nz]
        print(f"action {k}: nonzero on {len(nz)} steps (first {nz[0]['step']}, last {nz[-1]['step']}), mean {sum(v)/len(v):.3g}, max {max(v):.3g}, max mult {max(r['mult/'+k] for r in rows):.3g}")
    else:
        print(f"action {k}: never nonzero")
# first vs second step on the same batch, for the action terms
dk = ["step", "e", "eikonal", "consistency", "grid/one_piece_fraction", "grid/no_cavity_fraction", "grid/no_handle_fraction", "grid/handles_mean", "grid/clean_fraction",
      "graph_vs_grid/both_agreement", "graph_vs_grid/raw_extra_pieces", "graph_vs_grid/raw_cavities", "graph_vs_grid/confirm_dropped_kind0", "graph_vs_grid/confirm_dropped_cavities",
      "dead/live_mean", "dead/live_min", "dead/live_max", "dead/wobble_share_mean", "dead/pose_share_mean", "dead/top_singular_value_mean",
      "dead/subspace_stability_mean", "dead/subspace_stability_count", "dead/subspace_stability_chance",
      "descriptor_spread", "probe_field_distance", "grid/volume_fraction_mean", "grid/volume_fraction_std",
      "pose/centroid_offset_mean", "pose/centroid_offset_max", "pose/trace_mean", "pose/trace_rel_error_rms", "pose/trace_rel_error_max", "pose/offdiag_abs_max",
      "pose/centroid_sq_mean", "pose/scale_sq_mean", "pose/axes_sq_mean", "time_s"]
diags = [json.loads(l) for l in open(p + "/diagnostics.jsonl")]
sel = diags if len(diags) <= 8 else [d for d in diags if d["step"] in (0, 500, 1000, 2000, 3000, 4000, 6000, 8000, 10000)]
print("diagnostics at steps", [d["step"] for d in sel])
for k in dk:
    print(f"{k:42s}", "  ".join(f"{d.get(k, float('nan')):9.4g}" for d in sel))
