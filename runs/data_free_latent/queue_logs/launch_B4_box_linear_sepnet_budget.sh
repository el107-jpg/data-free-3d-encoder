#!/usr/bin/env bash
# B4: B3 plus a SEPARATE base network (code -> 64 tanh -> 4, last layer at zero; shares nothing with the
# side network or FiLM). Linear head, deviation budget 0.01 -> 0.2 over 500 updates, J4 at 5,000 warm
# start, 1,000 updates. Branch rounded-box-base, commit 05b384f. Controls: B3 and J6 at 6,000.
# Waits for B3 (pid 516907) to finish, then for no data-free trainer on the GPU.
set -euo pipefail
NAME=B4_J4_5000_box_linear_sepnet_budget02_1k
OUT=/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/$NAME
J4=/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/J4_d64_nocentre_procedural96_even_code_leak05_beta100_fourier3_5k
while kill -0 516907 2> /dev/null; do sleep 30; done
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
  --set base_shape=rounded_box --set base_head_form=linear --set base_head_net=separate \
  --set base_budget=0.2 --set base_budget_start=0.01 --set base_budget_steps=500 \
  --set diag_every=250 --set ckpt_every=100 --set log_every=25 --set diag_resolution=96 \
  >> "$OUT/train.log" 2>&1
