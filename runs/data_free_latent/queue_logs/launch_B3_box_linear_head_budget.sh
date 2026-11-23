#!/usr/bin/env bash
# B3: J4 at 5,000 warm-started into the rounded-box decoder with the LINEAR learned head (its output is
# the base's deviation from the sphere) and a deviation budget opening from 0.01 to 0.2 over 500 updates
# (adaptive constraint base_deviation). Branch rounded-box-base, commit 1cad02d. 1,000 updates.
# Control: J6_J4_continued_to_10k at 6,000. Waits for the GPU (no data-free trainer running).
set -euo pipefail
NAME=B3_J4_5000_box_linear_head_budget02_1k
OUT=/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/$NAME
J4=/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/J4_d64_nocentre_procedural96_even_code_leak05_beta100_fourier3_5k
while pgrep -f "train_data_free_latent.py --output-dir" > /dev/null; do sleep 30; done
mkdir -p "$OUT"
cd /home/eric-luo/Downloads/pointcloud-motion-planning
python3 runs/data_free_latent/analysis_scripts/scan_cache.py --evict .venv/lib/python3.12/site-packages \
  $HOME/.local/share/uv/python .claude/worktrees/rounded-box-base/arm_planner .claude/worktrees/rounded-box-base/scripts \
  scripts "$J4/checkpoint_last.pt" >> "$OUT/train.log" 2>&1 || true
ln -sfn "$OUT/tb" /home/eric-luo/Downloads/pointcloud-motion-planning/runs/tensorboard_active/data_free_$NAME
echo "started $(date)" >> "$OUT/train.log"
cd /home/eric-luo/Downloads/pointcloud-motion-planning/.claude/worktrees/rounded-box-base
PYTHONPATH=. exec /home/eric-luo/Downloads/pointcloud-motion-planning/.venv/bin/python /home/eric-luo/Downloads/pointcloud-motion-planning/scripts/gpu_capped.py --gpu-fraction 0.75 --gpu-util 0.7 \
  scripts/train_data_free_latent.py --output-dir "$OUT" --device cuda \
  --config-from "$J4/config.json" --steps 1000 --seed 0 \
  --set init_model_from=$J4/checkpoint_last.pt \
  --set base_shape=rounded_box --set base_head_form=linear \
  --set base_budget=0.2 --set base_budget_start=0.01 --set base_budget_steps=500 \
  --set diag_every=250 --set ckpt_every=100 --set log_every=25 --set diag_resolution=96 \
  >> "$OUT/train.log" 2>&1
