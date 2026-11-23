"""Weak-direction pair test against the control (the long run's own updates from 2,500).

usage: u_summary.py [run folder name] [window]
"""
import json
import sys

ROOT = "/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/"
run = sys.argv[1] if len(sys.argv) > 1 else "u_weakpairs_from2500"
width = int(sys.argv[2]) if len(sys.argv) > 2 else 50


def load(name, lo, hi):
    rows = {}
    for line in open(ROOT + name + "/metrics.jsonl"):
        r = json.loads(line)
        if lo <= r["step"] < hi:
            rows[r["step"]] = r
    diag = {}
    for line in open(ROOT + name + "/diagnostics.jsonl"):
        r = json.loads(line)
        if lo <= int(r["step"]) <= hi:
            diag[int(r["step"])] = r
    return [rows[k] for k in sorted(rows)], diag


test, tdiag = load(run, 2500, 10**9)
last = test[-1]["step"] if test else 2500
ctrl, cdiag = load("t_batchwide128", 2500, last + 1)
print(f"{run}: last update {last}")
if test and "weak/new_rms" in test[-1]:
    print("\nwindow | frame_term  new_rms (independent part)  new_raw_rms  used_rms_min")
    for a in range(2500, last + 1, width):
        rs = [r for r in test if a <= r["step"] < a + width]
        if rs:
            m = lambda k: sum(r.get(k, 0.0) for r in rs) / len(rs)
            print(f'{a:6d} | {m("weak/frame_term"):9.3f}  {m("weak/new_rms"):8.4f}  {m("weak/new_raw_rms"):8.4f}  {m("weak/used_rms_min"):8.4f}')
print("\nwindow | used  usage_eff  usage_top  usage_med | pair_rate  pair_min  pair_term | nearest (ctrl)   cons   cons_mult (ctrl) | thick_mult  walls (ctrl)  gnorm (ctrl) skipped")
for a in range(2500, last + 1, width):
    rs = [r for r in test if a <= r["step"] < a + width]
    cs = [r for r in ctrl if a <= r["step"] < a + width]
    if not rs:
        continue
    m = lambda k, s=rs: sum(r.get(k, 0.0) for r in s) / max(1, len(s))
    print(f'{a:6d} | {m("weak/used_directions"):4.1f}  {m("weak/usage_effective"):8.2f}  {m("weak/usage_top"):8.4f}  {m("weak/usage_median"):8.4f} | '
          f'{m("weak/pair_rate"):8.4f}  {m("weak/pair_rate_min"):7.4f}  {m("weak/pair_term"):8.3f} | '
          f'{m("div/nearest_distance"):.4f} ({m("div/nearest_distance", cs):.4f})  {m("term/consistency"):.4f}  {m("mult/consistency"):6.1f} ({m("mult/consistency", cs):6.1f}) | '
          f'{m("mult/thicken"):6.1f}  {m("graph/wall_sites"):5.1f} ({m("graph/wall_sites", cs):5.1f})  {m("grad_norm"):6.1f} ({m("grad_norm", cs):6.1f})  {sum(r.get("skipped", 0) for r in rs)}')
print("\nupdate | test: live  eff  clean  centroid  trace | control: live  eff  clean")
for s in sorted(set(tdiag) | set(cdiag)):
    t, c = tdiag.get(s), cdiag.get(s)
    g = lambda r, k: (r.get(k, float("nan")) if r else float("nan"))
    print(f'{s:6d} |       {g(t, "dead/live_mean"):4.1f}  {g(t, "dead/effective_directions_mean"):4.2f}  {g(t, "grid/clean_fraction"):4.2f}  '
          f'{g(t, "pose/centroid_offset_mean"):.4f}  {g(t, "pose/trace_mean"):.4f} |          '
          f'{g(c, "dead/live_mean"):4.1f}  {g(c, "dead/effective_directions_mean"):4.2f}  {g(c, "grid/clean_fraction"):4.2f}')
print("time per update (ms):", round(sum(r["time_ms/total"] for r in test[-50:]) / max(1, len(test[-50:]))),
      "| control:", round(sum(r["time_ms/total"] for r in ctrl[-50:]) / max(1, len(ctrl[-50:]))))
