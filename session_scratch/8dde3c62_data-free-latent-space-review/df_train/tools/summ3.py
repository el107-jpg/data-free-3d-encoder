import json, sys
p = sys.argv[1]
rows = [json.loads(l) for l in open(p + "/metrics.jsonl")]
diags = [json.loads(l) for l in open(p + "/diagnostics.jsonl")]
names = ("eikonal", "consistency", "barrier", "origin", "volume", "pose_centroid", "pose_scale", "pose_axes", "bridge", "delete", "fill", "widen", "thicken")
pts = [int(a) for a in sys.argv[2:]]
w = 50
losses = [r["loss"] for r in rows]
print("steps", len(rows), "skipped", sum(r["skipped"] for r in rows), "modes", {m: sum(r["graph_mode"] == m for r in rows) for m in ("off", "sign", "full")})
print("loss min %.4g; steps with loss < -0.01: %d" % (min(losses), sum(l < -0.01 for l in losses)))
keys = ["loss", "diversity", "div/mean_ratio", "div/dead_zone_fraction", "e", "grad_norm"] + ["term/" + k for k in names] + \
    ["graph/" + k for k in ("fragments_kind0", "fragments_kind1", "cavities", "gap_sites", "wall_sites", "confirm_dropped_kind0", "confirm_dropped_cavities")] + ["mult/" + k for k in names]
print("window mean over", w, "steps starting at", pts)
for k in keys:
    out = []
    for s in pts:
        seg = [r[k] for r in rows if s <= r["step"] < s + w]
        out.append(sum(seg) / len(seg) if seg else float('nan'))
    print(f"{k:32s}", "  ".join(f"{v:10.4g}" for v in out))
for k in ("barrier", "bridge", "delete", "fill", "widen", "thicken"):
    nz = [r for r in rows if r["term/" + k] > 0]
    v = [r["term/" + k] for r in nz]
    print(f"{k}: nonzero on {len(nz)} steps" + (f" (first {nz[0]['step']}, last {nz[-1]['step']}), mean {sum(v)/len(v):.3g}, max {max(v):.3g}, max mult {max(r['mult/'+k] for r in rows):.4g}" if nz else ""))
# per-500-step block: confirmed fragments etc.
print("block  frag0/build  cav/build  gap  wall  ratio  delete  mult_delete  mult_thicken  barrier  mult_barrier")
for s in range(0, len(rows), 500):
    seg = [r for r in rows if s <= r["step"] < s + 500]
    b = [r for r in seg if r["graph_built"]]
    m = lambda k, q=seg: sum(r[k] for r in q) / max(len(q), 1)
    print(f"{s:6d} {m('graph/fragments_kind0', b):8.2f} {m('graph/cavities', b):8.3f} {m('graph/gap_sites', b):7.1f} {m('graph/wall_sites', b):7.1f} {m('div/mean_ratio'):6.3f} {m('term/delete'):9.2e} {m('mult/delete'):9.1f} {m('mult/thicken'):9.1f} {m('term/barrier'):9.2e} {m('mult/barrier'):8.1f}")
dk = ["step", "e", "eikonal", "consistency", "grid/pieces_mean", "grid/one_piece_fraction", "grid/no_cavity_fraction", "grid/no_handle_fraction", "grid/clean_fraction",
      "graph_vs_grid/both_agreement", "graph_vs_grid/confirm_dropped_kind0", "graph_vs_grid/confirm_dropped_cavities",
      "dead/live_mean", "dead/live_min", "dead/live_max", "dead/wobble_share_mean", "dead/pose_share_mean", "dead/top_singular_value_mean",
      "dead/subspace_stability_mean", "dead/subspace_stability_chance", "descriptor_spread", "probe_field_distance",
      "pose/centroid_offset_mean", "pose/centroid_offset_max", "pose/trace_rel_error_rms", "pose/trace_rel_error_max", "pose/offdiag_abs_max"]
for chunk in (diags[:11], diags[11:]):
    if not chunk:
        continue
    for k in dk:
        print(f"{k:40s}", " ".join(f"{d.get(k, float('nan')):8.3g}" for d in chunk))
    print()
print("diagnostics all clean:", sum(d["grid/clean_fraction"] == 1.0 for d in diags), "of", len(diags))
