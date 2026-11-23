#!/usr/bin/env bash
# B10: B9 (base spread push) continued from 3,000 to 5,000 with the learned base head's learning rate at 0.1x
# (base_lr_scale=0.1), so the correction can catch up with the base. Spread targets and budget as in B9.
# Branch rounded-box-base. Controls: B9 at 3,000 (start), B7 at 5,000 (no regulariser).
set -uo pipefail
OUT=/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/B10_B9_cont_baselr01_to5k
cd /home/eric-luo/Downloads/pointcloud-motion-planning
while pgrep -f "train_data_free_latent[.]py --output-dir" > /dev/null; do sleep 30; done
python3 runs/data_free_latent/analysis_scripts/scan_cache.py --evict .venv/lib/python3.12/site-packages \
  $HOME/.local/share/uv/python .claude/worktrees/rounded-box-base/arm_planner .claude/worktrees/rounded-box-base/scripts \
  scripts "$OUT" >> "$OUT/train.log" 2>&1 || true
ln -sfn "$OUT/tb" /home/eric-luo/Downloads/pointcloud-motion-planning/runs/tensorboard_active/data_free_B10_B9_cont_baselr01_to5k
echo "started $(date)" >> "$OUT/train.log"
( until [ -f "$OUT/metrics.jsonl" ] && [ "$(tail -1 "$OUT/metrics.jsonl" | python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["step"])' 2>/dev/null || echo 0)" -ge 4000 ]; do sleep 60; kill -0 $$ 2>/dev/null || exit 0; done
  cp "$OUT/checkpoint_last.pt" "$OUT/checkpoint_4000.pt" ) &
cd /home/eric-luo/Downloads/pointcloud-motion-planning/.claude/worktrees/rounded-box-base
PYTHONPATH=. /home/eric-luo/Downloads/pointcloud-motion-planning/.venv/bin/python /home/eric-luo/Downloads/pointcloud-motion-planning/scripts/gpu_capped.py --gpu-fraction 0.75 --gpu-util 0.7 \
  scripts/train_data_free_latent.py --output-dir "$OUT" --device cuda --resume --steps 5000 --set base_lr_scale=0.1 \
  >> "$OUT/train.log" 2>&1
echo "finished (exit $?) $(date)" >> "$OUT/train.log"
