#!/usr/bin/env bash
# B1: J4 at 5,000 warm-started into the rounded-box decoder (branch rounded-box-base), 1,000 updates.
# Control: J6_J4_continued_to_10k at 6,000 (J4 continued with nothing changed). Waits for the GPU first.
set -euo pipefail
NAME=B1_J4_5000_box_base_ramp200_1k
OUT=/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/$NAME
J4=/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/J4_d64_nocentre_procedural96_even_code_leak05_beta100_fourier3_5k
while pgrep -f train_data_free_latent.py > /dev/null; do sleep 30; done
mkdir -p "$OUT"
ln -sfn "$OUT/tb" /home/eric-luo/Downloads/pointcloud-motion-planning/runs/tensorboard_active/data_free_$NAME
echo "started $(date)" >> "$OUT/train.log"
cd /home/eric-luo/Downloads/pointcloud-motion-planning/.claude/worktrees/rounded-box-base
PYTHONPATH=. exec /home/eric-luo/Downloads/pointcloud-motion-planning/.venv/bin/python /home/eric-luo/Downloads/pointcloud-motion-planning/scripts/gpu_capped.py --gpu-fraction 0.75 --gpu-util 0.7 \
  scripts/train_data_free_latent.py --output-dir "$OUT" --device cuda \
  --config-from "$J4/config.json" --steps 1000 --seed 0 \
  --set init_model_from=$J4/checkpoint_last.pt \
  --set base_shape=rounded_box --set base_ramp_steps=200 \
  --set diag_every=250 --set ckpt_every=100 --set log_every=25 --set diag_resolution=96 \
  >> "$OUT/train.log" 2>&1
