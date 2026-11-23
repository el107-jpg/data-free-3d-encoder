#!/bin/bash
# Start a from-scratch anchor run once another run has finished.
# usage: queue_run.sh <run to wait for, or - for none> <new run name> <updates> [--set key=value ...]
# The base settings are those of the batch-wide run plus the 64 anchor objects; later --set
# arguments override them.
M=/home/eric-luo/Downloads/pointcloud-motion-planning
WAIT=$1; NAME=$2; STEPS=$3; shift 3
if [ "$WAIT" != "-" ]; then
  # the other run must have started (its log exists) and its trainer must be gone
  until [ -e "$M/runs/data_free_latent/$WAIT/train.log" ] && ! pgrep -f "train_data_free_latent.py --output-dir .*/$WAIT " > /dev/null; do
    sleep 20
  done
  sleep 5
fi
O=$M/runs/data_free_latent/$NAME
mkdir -p "$O"
ln -sfn "$O/tb" "$M/runs/tensorboard_active/data_free_$NAME"
cd /home/eric-luo/Downloads/pointcloud-motion-planning/.claude/worktrees/data-free-connectivity-plan || exit 1
PYTHONPATH=. $M/.venv/bin/python $M/scripts/gpu_capped.py --gpu-fraction 0.75 --gpu-util 0.7 \
  scripts/train_data_free_latent.py --output-dir "$O" --steps "$STEPS" --device cuda --seed 0 \
  --set accumulate=4 --set lr=6.364e-4 --set lr_warmup_steps=125 --set lr_floor_fraction=1.0 \
  --set mult_rate=0.08 --set mult_rates.bridge=0.2 --set mult_rates.delete=0.2 --set mult_rates.fill=0.2 \
  --set mult_rates.widen=0.2 --set mult_rates.thicken=0.2 --set mult_rates.barrier=0.2 \
  --set diag_every=125 --set ckpt_every=100 --set log_every=25 --set diag_resolution=96 \
  --set clusters=4 --set codes_per_cluster=8 --set global_codes=True --set diversity_mode=nearest \
  --set spread_weight=0.0 --set batch_neighbours=True --set pool_size=0 --set pool_points=1024 --set nearest_k=8 \
  --set anchor_path=$M/datasets/data_free_anchors_20261111_v1/anchors.pt --set anchor_codes_per_pass=4 \
  --set anchor_weight=100.0 --set anchor_surface_weight=1.0 --set anchor_embed_lr=3e-4 \
  --set anchor_freeze_steps=500 --set anchor_ramp_steps=1000 "$@" > "$O/train.log" 2>&1
echo "$NAME finished: $(grep -E '^diag' "$O/train.log" | tail -n 1 | cut -c1-260)"
