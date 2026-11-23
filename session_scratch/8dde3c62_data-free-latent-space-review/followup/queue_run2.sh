#!/bin/bash
# Start a from-scratch anchor run (the "L2" settings) once another run has finished.
# usage: [SEED=n] [ANCHORS=path] queue_run2.sh <run to wait for, or -> <new run name> <updates> [--set key=value ...]
# Base settings: batch-wide run + 64 anchors, pull 100, neighbourhood radius 0.5, anchors exempt from the
# field-accuracy constraints, codes frozen 500 updates then ramped to 3e-4 by 1,500. Later --set arguments override.
M=/home/eric-luo/Downloads/pointcloud-motion-planning
SEED=${SEED:-0}
ANCHORS=${ANCHORS:-$M/datasets/data_free_anchors_20261111_v1/anchors.pt}
WAIT=$1; NAME=$2; STEPS=$3; shift 3
if [ "$WAIT" != "-" ]; then
  until [ -e "$M/runs/data_free_latent/$WAIT/train.log" ] && ! pgrep -f "train_data_free_latent.py --output-dir .*/$WAIT " > /dev/null; do
    sleep 20
  done
  sleep 5
fi
O=$M/runs/data_free_latent/$NAME
if [ -e "$O/train.log" ]; then echo "$NAME already exists; refusing to overwrite"; exit 1; fi
mkdir -p "$O"
ln -sfn "$O/tb" "$M/runs/tensorboard_active/data_free_$NAME"
cd /home/eric-luo/Downloads/pointcloud-motion-planning/.claude/worktrees/data-free-connectivity-plan || exit 1
PYTHONPATH=. $M/.venv/bin/python $M/scripts/gpu_capped.py --gpu-fraction 0.75 --gpu-util 0.7 \
  scripts/train_data_free_latent.py --output-dir "$O" --steps "$STEPS" --device cuda --seed "$SEED" \
  --set accumulate=4 --set lr=6.364e-4 --set lr_warmup_steps=125 --set lr_floor_fraction=1.0 \
  --set mult_rate=0.08 --set mult_rates.bridge=0.2 --set mult_rates.delete=0.2 --set mult_rates.fill=0.2 \
  --set mult_rates.widen=0.2 --set mult_rates.thicken=0.2 --set mult_rates.barrier=0.2 \
  --set diag_every=250 --set ckpt_every=100 --set log_every=25 --set diag_resolution=96 \
  --set clusters=4 --set codes_per_cluster=8 --set global_codes=True --set diversity_mode=nearest \
  --set spread_weight=0.0 --set batch_neighbours=True --set pool_size=0 --set pool_points=1024 --set nearest_k=8 \
  --set anchor_path=$ANCHORS --set anchor_codes_per_pass=4 \
  --set anchor_weight=100.0 --set anchor_surface_weight=1.0 --set anchor_embed_lr=3e-4 \
  --set anchor_freeze_steps=500 --set anchor_ramp_steps=1000 \
  --set anchor_radius=0.5 --set anchor_exempt_field=True "$@" > "$O/train.log" 2>&1
echo "$NAME finished: $(grep -E '^diag' "$O/train.log" | tail -n 1 | cut -c1-260)"
