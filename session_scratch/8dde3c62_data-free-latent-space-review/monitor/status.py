#!/usr/bin/env python3
"""Live status of the data-free latent training runs. Standard library only; reads, never writes.

usage: python3 status.py            everything: what is training, what is queued, recent finished runs
       python3 status.py <run>      more detail on one run folder (trend over the run)

It looks at the running processes (trainer, queue scripts, anchor build) and at each run folder's
metrics.jsonl / diagnostics.jsonl / train.log under runs/data_free_latent/.
"""
import json
import os
import subprocess
import sys
import time

ROOT = "/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent"
NOTE = os.path.join(ROOT, "monitor", "RUNS.md")


def tail_rows(path, size=400_000):
    """Last rows of a .jsonl file without reading all of it."""
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            end = fh.tell()
            fh.seek(max(0, end - size))
            text = fh.read().decode("utf-8", "ignore")
    except OSError:
        return []
    rows = []
    for line in text.splitlines()[1 if end > size else 0:]:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def processes():
    out = subprocess.run(["ps", "-eo", "pid,etimes,args"], capture_output=True, text=True).stdout.splitlines()[1:]
    found = []
    for line in out:
        parts = line.split(None, 2)
        if len(parts) < 3 or "bash -c" in parts[2] or "status.py" in parts[2]:
            continue
        found.append((int(parts[0]), int(parts[1]), parts[2]))
    return found


def arg_after(args, flag):
    words = args.split()
    return words[words.index(flag) + 1] if flag in words else None


def settings(args):
    words = args.split()
    return [words[i + 1] for i, w in enumerate(words[:-1]) if w == "--set"]


def mean(rows, key):
    vals = [r[key] for r in rows if key in r and isinstance(r[key], (int, float))]
    return sum(vals) / len(vals) if vals else float("nan")


def fmt_time(seconds):
    seconds = int(max(0, seconds))
    return f"{seconds // 3600} h {seconds % 3600 // 60:02d} min" if seconds >= 3600 else f"{seconds // 60} min"


def run_summary(name, target=None, window=100):
    folder = os.path.join(ROOT, name)
    rows = tail_rows(os.path.join(folder, "metrics.jsonl"))
    diag = tail_rows(os.path.join(folder, "diagnostics.jsonl"), 60_000)
    lines = []
    if not rows:
        return ["  (no updates logged yet)"]
    step = rows[-1]["step"] + 1
    last = rows[-window:]
    per = mean(last, "time_ms/total") / 1000.0
    head = f"  update {step}" + (f" of {target}" if target else "")
    if target and per == per:
        head += f"; about {per:.1f} s per update; roughly {fmt_time((target - step) * per)} left"
    lines.append(head)
    age = time.time() - os.path.getmtime(os.path.join(folder, "metrics.jsonl"))
    if age > 300:
        lines.append(f"  WARNING: nothing logged for {fmt_time(age)} (stalled or stopped?)")
    lines.append(f"  last {len(last)} updates: nearest-neighbour distance {mean(last, 'div/nearest_distance'):.4f}; "
                 f"consistency {mean(last, 'term/consistency'):.4f} (tolerance {mean(last, 'tol/consistency'):.4f}, "
                 f"multiplier {mean(last, 'mult/consistency'):.0f}); eikonal {mean(last, 'term/eikonal'):.5f} "
                 f"(multiplier {mean(last, 'mult/eikonal'):.0f})")
    lines.append(f"    thin-wall sites per pass {mean(last, 'graph/wall_sites'):.1f}; gap sites {mean(last, 'graph/gap_sites'):.2f}; "
                 f"thicken multiplier {mean(last, 'mult/thicken'):.1f}; gradient size {mean(last, 'grad_norm'):.0f}; "
                 f"skipped updates {int(sum(r.get('skipped', 0) for r in last))}")
    if "anchor/iou" in rows[-1]:
        lines.append(f"    anchor overlap in training {mean(last, 'anchor/iou'):.2f}; anchor distance {mean(last, 'anchor/distance'):.4f}")
    if diag:
        d = diag[-1]
        g = lambda k: d.get(k, float("nan"))
        lines.append(f"  last check (update {int(d['step'])}): probes clean {g('grid/clean_fraction'):.2f} "
                     f"(one piece {g('grid/one_piece_fraction'):.2f}, no handle {g('grid/no_handle_fraction'):.2f}); "
                     f"live directions {g('dead/live_mean'):.1f}; effective {g('dead/effective_directions_mean'):.1f}; "
                     f"wobble {g('dead/wobble_share_mean'):.2f}; probe distance {g('probe_field_distance'):.4f}"
                     + (f"; anchors reproduced {g('anchor/iou_mean'):.2f} (worst {g('anchor/iou_min'):.2f})"
                        if "anchor/iou_mean" in d else ""))
    return lines


def trend(name):
    folder = os.path.join(ROOT, name)
    diag = tail_rows(os.path.join(folder, "diagnostics.jsonl"), 4_000_000)
    print(f"{name}: periodic checks")
    print(" update  clean  live  effective  probe-distance  anchors(worst)")
    for d in diag[-24:]:
        g = lambda k: d.get(k, float("nan"))
        print(f" {int(d['step']):6d}  {g('grid/clean_fraction'):5.2f}  {g('dead/live_mean'):4.1f}  "
              f"{g('dead/effective_directions_mean'):8.1f}  {g('probe_field_distance'):13.4f}  "
              f"{g('anchor/iou_mean'):.2f} ({g('anchor/iou_min'):.2f})")
    cfg_path = os.path.join(folder, "config.json")
    if os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path))
        keys = ("latent_dim", "anchor_path", "anchor_count", "anchor_weight", "anchor_radius", "anchor_embed_lr",
                "anchor_exempt_field", "consistency_fixed_surface", "consistency_min_depth", "consistency_tolerance_start",
                "correction_bound_inner", "origin_margin", "weak_frame_passes", "seed", "steps", "lr")
        print("settings:", {k: cfg.get(k) for k in keys if k in cfg}, "| consistency tolerance:", cfg.get("tolerances", {}).get("consistency"))
    print("\n".join(run_summary(name)))


def main():
    if len(sys.argv) > 1:
        trend(sys.argv[1])
        return
    print(time.strftime("%Y-%m-%d %H:%M %Z"))
    procs = processes()
    running = set()
    print("\nTRAINING NOW")
    any_training = False
    for pid, age, args in procs:
        if "scripts/train_data_free_latent.py" in args and "gpu_capped.py" in args:
            any_training = True
            out = arg_after(args, "--output-dir")
            name = os.path.basename(out.rstrip("/")) if out else "?"
            running.add(name)
            target = arg_after(args, "--steps")
            print(f"- {name} (running {fmt_time(age)}{', resumed' if '--resume' in args.split() else ''})")
            print("  settings changed from the base:", ", ".join(settings(args)[-8:]) or "none")
            print("\n".join(run_summary(name, int(target) if target else None)))
    if not any_training:
        print("- nothing is training")
    print("\nQUEUED (each starts when the run it waits for ends)")
    queued = False
    for pid, age, args in procs:
        for script in ("queue_run2.sh", "queue_run.sh", "cont_run.sh"):
            if script in args:
                words = args.split()
                rest = words[words.index(next(w for w in words if w.endswith(script))) + 1:]
                new = rest[2] if script == "cont_run.sh" else rest[1]
                if new in running:
                    continue
                queued = True
                print(f"- {new}: waits for {rest[0]}; {' '.join(rest[3:] if script != 'cont_run.sh' else rest[3:])[:200]}")
    if not queued:
        print("- nothing is queued")
    for pid, age, args in procs:
        if "build_data_free_anchors.py" in args:
            print(f"\nANCHOR BUILD running for {fmt_time(age)}: output {arg_after(args, '--output-dir')}")
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw",
                              "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
        print(f"\nGPU: {gpu}")
    except OSError:
        pass
    print("\nRECENTLY FINISHED OR STOPPED (newest first)")
    folders = [d for d in os.listdir(ROOT) if os.path.exists(os.path.join(ROOT, d, "metrics.jsonl")) and d not in running]
    folders.sort(key=lambda d: os.path.getmtime(os.path.join(ROOT, d, "metrics.jsonl")), reverse=True)
    for name in folders[:5]:
        rows = tail_rows(os.path.join(ROOT, name, "metrics.jsonl"), 20_000)
        when = time.strftime("%m-%d %H:%M", time.localtime(os.path.getmtime(os.path.join(ROOT, name, "metrics.jsonl"))))
        print(f"- {name}: last update {rows[-1]['step'] + 1 if rows else '?'}, last written {when}")
    print(f"\nNotes on what each run is for and results so far: {NOTE}")


if __name__ == "__main__":
    main()
