"""Continuation of the batch-wide run: windowed state from a given update on.

usage: t_cont.py [start] [width]
"""
import json
import sys

T = "/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/t_batchwide128/"
start = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
width = int(sys.argv[2]) if len(sys.argv) > 2 else 250
rows = {}
for line in open(T + "metrics.jsonl"):
    r = json.loads(line)
    rows[r["step"]] = r                      # a resumed run may rewrite a step; keep the latest
rows = [rows[k] for k in sorted(rows)]
diag = {}
for line in open(T + "diagnostics.jsonl"):
    r = json.loads(line)
    diag[int(r["step"])] = r
print("diagnostic keys with pose/centroid/trace:", [k for k in diag[max(diag)] if any(s in k for s in ("centroid", "trace", "pose", "size"))][:12])
print("\nupdate  live  eff  clean  no_handle  one_piece | keys of interest")
for s in sorted(diag):
    if s < start:
        continue
    r = diag[s]
    extra = " ".join(f'{k.split("/")[-1]}={r[k]:.4f}' for k in r if any(t in k for t in ("centroid", "trace")) and isinstance(r[k], (int, float)))
    print(f'{s:6d}  {r.get("dead/live_mean", float("nan")):4.1f}  {r.get("dead/effective_directions_mean", float("nan")):4.2f}  '
          f'{r.get("grid/clean_fraction", float("nan")):5.2f}  {r.get("grid/no_handle_fraction", float("nan")):5.2f}  '
          f'{r.get("grid/one_piece_fraction", float("nan")):5.2f} | {extra}')
print("\nwindow  nearest  cons(min..max)        cons_mult(min..max)  thicken   thick_mult  widen_mult delete_mult  pose_c    pose_s    walls  gaps  frag1  gnorm  skipped")
for a in range(start, rows[-1]["step"] + 1, width):
    rs = [r for r in rows if a <= r["step"] < a + width]
    if not rs:
        continue
    m = lambda k: sum(r.get(k, 0.0) for r in rs) / len(rs)
    lo = lambda k: min(r.get(k, 0.0) for r in rs)
    hi = lambda k: max(r.get(k, 0.0) for r in rs)
    print(f'{a:6d}  {m("div/nearest_distance"):.4f}  {m("term/consistency"):.4f} ({lo("term/consistency"):.4f}..{hi("term/consistency"):.4f})  '
          f'{m("mult/consistency"):6.1f} ({lo("mult/consistency"):5.1f}..{hi("mult/consistency"):5.1f})  {m("term/thicken"):.1e}  {m("mult/thicken"):7.1f}   '
          f'{m("mult/widen"):7.1f}   {m("mult/delete"):7.1f}   {m("term/pose_centroid"):.1e}  {m("term/pose_scale"):.1e}  '
          f'{m("graph/wall_sites"):5.1f} {m("graph/gap_sites"):5.2f} {m("graph/fragments_kind1"):5.2f} {m("grad_norm"):6.1f}  {sum(r.get("skipped", 0) for r in rs)}')
print("last update", rows[-1]["step"])
