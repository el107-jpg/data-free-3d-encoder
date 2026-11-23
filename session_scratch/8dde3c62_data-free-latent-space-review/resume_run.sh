#!/bin/bash
# Resume an interrupted run in its OWN folder from its latest checkpoint (the log is appended to).
# usage: resume_run.sh <run name> <total updates> [--set key=value ...]
# Pass the same --set arguments the run was started with if they were given on a continuation.
M=/home/eric-luo/Downloads/pointcloud-motion-planning
NAME=$1; STEPS=$2; shift 2
O=$M/runs/data_free_latent/$NAME
if [ ! -e "$O/checkpoint_last.pt" ]; then echo "$NAME has no checkpoint to resume from"; exit 1; fi
if pgrep -f "train_data_free_latent.py --output-dir .*/$NAME " > /dev/null; then echo "$NAME is already training"; exit 1; fi
cd /home/eric-luo/Downloads/pointcloud-motion-planning/.claude/worktrees/data-free-connectivity-plan || exit 1
echo "--- resumed $(date '+%Y-%m-%d %H:%M:%S') ---" >> "$O/train.log"
PYTHONPATH=. $M/.venv/bin/python $M/scripts/gpu_capped.py --gpu-fraction 0.75 --gpu-util 0.7 \
  scripts/train_data_free_latent.py --output-dir "$O" --steps "$STEPS" --device cuda --resume \
  --set ckpt_every=100 "$@" >> "$O/train.log" 2>&1
echo "$NAME finished: $(grep -E '^diag' "$O/train.log" | tail -n 1 | cut -c1-260)"
