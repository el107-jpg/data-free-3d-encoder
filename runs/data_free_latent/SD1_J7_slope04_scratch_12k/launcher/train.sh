#!/usr/bin/env bash
set -eu
run_dir=$(cd "$(dirname "$0")/.." && pwd)
repo=/home/eric-luo/Downloads/pointcloud-motion-planning
export PYTHONPATH="$repo/.claude/worktrees/j7-slope-diversity"
export CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=3 OMP_NUM_THREADS=3 PYTHONUNBUFFERED=1
cd "$PYTHONPATH"
exec "$repo/.venv/bin/python" "$run_dir/launcher/worker.py" "$run_dir" >>"$run_dir/train.log" 2>&1
