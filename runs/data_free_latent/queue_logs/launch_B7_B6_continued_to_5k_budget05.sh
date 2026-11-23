#!/usr/bin/env bash
# B7: the long test of the ablation winner. B6 (from scratch; learned rounded-box base, linear head form,
# base head = side network features -> 64 tanh -> 4) resumed at 1,000 and continued to 5,000. The deviation
# budget is widened: final 0.5 reached at update 2,500 (0.206 at 1,000, continuous with B6's 0.2).
# Branch rounded-box-base, commit 4aa1deb. Control: J4 at the same updates. Snapshots at 2,000/3,000/4,000.
set -euo pipefail
NAME=B7_B6_continued_to_5k_budget05
OUT=/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/$NAME
while pgrep -f "train_data_free_latent.py --output-dir" > /dev/null; do sleep 30; done
cd /home/eric-luo/Downloads/pointcloud-motion-planning
python3 runs/data_free_latent/analysis_scripts/scan_cache.py --evict .venv/lib/python3.12/site-packages \
  $HOME/.local/share/uv/python .claude/worktrees/rounded-box-base/arm_planner .claude/worktrees/rounded-box-base/scripts \
  scripts "$OUT/checkpoint_last.pt" >> "$OUT/train.log" 2>&1 || true
ln -sfn "$OUT/tb" /home/eric-luo/Downloads/pointcloud-motion-planning/runs/tensorboard_active/data_free_$NAME
echo "started $(date)" >> "$OUT/train.log"
( for T in 2000 3000 4000; do
    until [ -f "$OUT/metrics.jsonl" ] && [ "$(tail -1 "$OUT/metrics.jsonl" | python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["step"])' 2>/dev/null || echo 0)" -ge $T ]; do sleep 60; done
    cp "$OUT/checkpoint_last.pt" "$OUT/checkpoint_$T.pt"
  done ) &
cd /home/eric-luo/Downloads/pointcloud-motion-planning/.claude/worktrees/rounded-box-base
PYTHONPATH=. /home/eric-luo/Downloads/pointcloud-motion-planning/.venv/bin/python /home/eric-luo/Downloads/pointcloud-motion-planning/scripts/gpu_capped.py --gpu-fraction 0.75 --gpu-util 0.7 \
  scripts/train_data_free_latent.py --output-dir "$OUT" --device cuda --resume --steps 5000 \
  --set base_budget=0.5 --set base_budget_steps=2500 \
  >> "$OUT/train.log" 2>&1
