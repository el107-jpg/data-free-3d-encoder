import json, sys
ROOT = "/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/"
runs = sys.argv[1:]
for run in runs:
    rows = [json.loads(l) for l in open(f"{ROOT}{run}/diagnostics.jsonl")]
    print("==", run, "keys:", len(rows[0]))
    if run == runs[0]:
        print(sorted(rows[0].keys()))
    cols = ["step", "dead/live_mean", "dead/effective_directions_mean", "dead/wobble_share_mean", "dead/subspace_stability_mean", "dead/subspace_stability_chance",
            "grid/clean_fraction", "grid/no_handle_fraction", "grid/one_piece_fraction", "descriptor_spread", "probe_field_distance",
            "pose/centroid_offset_mean", "pose/trace_rel_error_rms", "pose/offdiag_abs_max", "grid/volume_fraction_std", "descriptor_std/elongation", "descriptor_std/flatness", "descriptor_std/sphericity"]
    cols = [c for c in cols if c in rows[-1]]
    print(" | ".join(c.split("/")[-1][:14] for c in cols))
    for r in rows:
        print(" | ".join(f"{r.get(c, float('nan')):14.4g}" for c in cols))
