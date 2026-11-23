#!/bin/bash
# Continue a saved model in a NEW run folder (the source checkpoint is copied, never modified).
# usage: cont_run.sh <run to wait for, or -> <source checkpoint> <new run name> <total updates> [--set key=value ...]
M=/home/eric-luo/Downloads/pointcloud-motion-planning
WAIT=$1; SRC=$2; NAME=$3; STEPS=$4; shift 4
if [ "$WAIT" != "-" ]; then
  until [ -e "$M/runs/data_free_latent/$WAIT/train.log" ] && ! pgrep -f "train_data_free_latent.py --output-dir .*/$WAIT " > /dev/null; do
    sleep 20
  done
  sleep 5
fi
O=$M/runs/data_free_latent/$NAME
if [ -e "$O/checkpoint_last.pt" ]; then echo "$NAME already exists; refusing to overwrite"; exit 1; fi
mkdir -p "$O"
cp "$SRC" "$O/checkpoint_last.pt"
cp "$SRC" "$O/checkpoint_start.pt"
ln -sfn "$O/tb" "$M/runs/tensorboard_active/data_free_$NAME"
cd /home/eric-luo/Downloads/pointcloud-motion-planning/.claude/worktrees/data-free-connectivity-plan || exit 1
PYTHONPATH=. $M/.venv/bin/python $M/scripts/gpu_capped.py --gpu-fraction 0.75 --gpu-util 0.7 \
  scripts/train_data_free_latent.py --output-dir "$O" --steps "$STEPS" --device cuda --resume \
  --set ckpt_every=100 "$@" > "$O/train.log" 2>&1
echo "$NAME finished: $(grep -E '^diag' "$O/train.log" | tail -n 1 | cut -c1-260)"
