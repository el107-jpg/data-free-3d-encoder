#!/usr/bin/env bash
# B17 resume after the segfault at 10,735 (RAM fault): 10,700 -> 11,000, same settings (stored in the checkpoint).
set -uo pipefail
OUT=/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/B17_B12_cont_bend_taper_repel03_bendscale01_fulllr_to11k
cd /home/eric-luo/Downloads/pointcloud-motion-planning
python3 runs/data_free_latent/analysis_scripts/scan_cache.py --evict .venv/lib/python3.12/site-packages \
  $HOME/.local/share/uv/python .claude/worktrees/rounded-box-base/arm_planner .claude/worktrees/rounded-box-base/scripts \
  scripts "$OUT" >> "$OUT/train.log" 2>&1 || true
echo "resumed after segfault $(date)" >> "$OUT/train.log"
cd /home/eric-luo/Downloads/pointcloud-motion-planning/.claude/worktrees/rounded-box-base
PYTHONPATH=. /home/eric-luo/Downloads/pointcloud-motion-planning/.venv/bin/python /home/eric-luo/Downloads/pointcloud-motion-planning/scripts/gpu_capped.py --gpu-fraction 0.75 --gpu-util 0.7 \
  scripts/train_data_free_latent.py --output-dir "$OUT" --device cuda --resume --steps 11000 >> "$OUT/train.log" 2>&1
echo "finished (exit $?) $(date)" >> "$OUT/train.log"
