#!/usr/bin/env bash
# B2: J4 at 5,000 warm-started into the rounded-box decoder with the FIXED code-to-base map and
# anchor-fitted base components (branch rounded-box-base, commit 73b3107); bases ramped in over
# 500 updates, 2,000 updates in all. Control: J6_J4_continued_to_10k at 7,000. Waits for the GPU.
set -euo pipefail
NAME=B2_J4_5000_box_fixed_map_anchorfit_ramp500_2k
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
  --config-from "$J4/config.json" --steps 2000 --seed 0 \
  --set init_model_from=$J4/checkpoint_last.pt \
  --set base_shape=rounded_box --set base_map=fixed --set anchor_base_fit=True --set base_ramp_steps=500 \
  --set diag_every=250 --set ckpt_every=100 --set log_every=25 --set diag_resolution=96 \
  >> "$OUT/train.log" 2>&1
