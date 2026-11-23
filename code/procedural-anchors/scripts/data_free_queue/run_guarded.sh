#!/bin/bash
# Run one training command, guarding against the machine's memory fault (Nov 2026: copies of files held in
# memory go bad while the disk stays clean, and a new Python process then dies on import).
# usage: run_guarded.sh <run folder> <worktree> <tag> -- <trainer arguments ...>
# <tag> is not used here: the launchers pass "train_data_free_latent.py --output-dir <run folder> " so that
# this process, too, matches the pattern every waiting script looks for. A stage that is being scanned or
# retried then still counts as running, and nothing queued behind it starts early.
# Before each attempt the Python environment, the Python installation, this worktree's code, the anchor
# sets and the run folder are compared with the disk and damaged in-memory copies are dropped. If the
# trainer exits without having logged a single update, the log is kept as train_failed_<n>.log and it is
# tried again, up to three attempts. A trainer that got as far as training is never restarted here.
M=/home/eric-luo/Downloads/pointcloud-motion-planning
O=$1; W=$2; shift 4
for attempt in 1 2 3; do
  python3 $M/runs/data_free_latent/analysis_scripts/scan_cache.py --evict $M/.venv \
    $HOME/.local/share/uv/python/cpython-3.12.16-linux-x86_64-gnu "$W/arm_planner" "$W/scripts" \
    $M/datasets/data_free_anchors_* "$O" > "$O/memory_scan_before_attempt_$attempt.txt" 2>&1
  cd "$W" || exit 1
  PYTHONPATH=. $M/.venv/bin/python $M/scripts/gpu_capped.py --gpu-fraction 0.75 --gpu-util 0.7 \
    scripts/train_data_free_latent.py --output-dir "$O" "$@" > "$O/train.log" 2>&1
  if grep -q -E "^step " "$O/train.log"; then exit 0; fi
  cp "$O/train.log" "$O/train_failed_$attempt.log"
  sleep 20
done
exit 1
