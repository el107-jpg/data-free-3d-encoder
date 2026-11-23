"""Trajectory of the batch-wide run: periodic diagnostics and windowed training terms."""
import json

T = "/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/t_batchwide128/"
d = [json.loads(l) for l in open(T + "diagnostics.jsonl")]
print("update  live  eff_dirs  top_share  fifth  clean  one_piece  no_handle  wobble  pose_share")
for r in d:
    if int(r["step"]) % 250:
        continue
    g = lambda k, default=float("nan"): r.get(k, default)
    print(f'{int(r["step"]):6d}  {g("dead/live_mean"):4.1f}  {g("dead/effective_directions_mean"):7.2f}  '
          f'{g("dead/top_direction_share_mean"):8.2f}  {g("dead/fifth_direction_motion_mean"):6.3f}  '
          f'{g("grid/clean_fraction"):5.2f}  {g("grid/one_piece_fraction"):8.2f}  {g("grid/no_handle_fraction"):8.2f}  '
          f'{g("dead/wobble_share_mean"):6.2f}  {g("dead/pose_share_mean"):6.2f}')
rows = [json.loads(l) for l in open(T + "metrics.jsonl")]
names = ["eikonal", "consistency", "barrier", "pose_centroid", "pose_scale", "pose_axes", "delete", "widen", "thicken"]
print("\nwindow  nearest  eik     cons    | multipliers: " + " ".join(n[:9] for n in names) + " | walls frag gnorm skipped")
for a in range(0, rows[-1]["step"] + 1, 250):
    rs = [r for r in rows if a <= r["step"] < a + 250]
    if not rs:
        continue
    m = lambda k: sum(r[k] for r in rs) / len(rs)
    print(f'{a:6d}  {m("div/nearest_distance"):.4f}  {m("term/eikonal"):.5f} {m("term/consistency"):.5f} | '
          + " ".join(f'{m("mult/" + n):9.1f}' for n in names)
          + f' | {m("graph/wall_sites"):5.1f} {m("graph/fragments_kind0"):4.2f} {m("grad_norm"):6.1f} {sum(r["skipped"] for r in rs)}')
print("last update", rows[-1]["step"])
