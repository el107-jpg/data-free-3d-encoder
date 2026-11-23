#!/usr/bin/env bash
# B12: B11 continued from 7,000 to 9,000 with the correction-size cost raised from 25 to 50
# (band 0.05). Everything else as B11 and B10
# (base head at 0.1x learning rate, spread push, budget). Branch rounded-box-base, commit see git log. Control: B11 at 7,000.
set -uo pipefail
OUT=/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/B12_B11_cont_corrcost50_to9k
cd /home/eric-luo/Downloads/pointcloud-motion-planning
while pgrep -f "train_data_free_latent[.]py --output-dir" > /dev/null; do sleep 30; done
python3 runs/data_free_latent/analysis_scripts/scan_cache.py --evict .venv/lib/python3.12/site-packages \
  $HOME/.local/share/uv/python .claude/worktrees/rounded-box-base/arm_planner .claude/worktrees/rounded-box-base/scripts \
  scripts "$OUT" >> "$OUT/train.log" 2>&1 || true
ln -sfn "$OUT/tb" /home/eric-luo/Downloads/pointcloud-motion-planning/runs/tensorboard_active/data_free_B12_B11_cont_corrcost50_to9k
echo "started $(date)" >> "$OUT/train.log"
( until [ -f "$OUT/metrics.jsonl" ] && [ "$(tail -1 "$OUT/metrics.jsonl" | python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["step"])' 2>/dev/null || echo 0)" -ge 8000 ]; do sleep 60; kill -0 $$ 2>/dev/null || exit 0; done
  cp "$OUT/checkpoint_last.pt" "$OUT/checkpoint_8000.pt" ) &
cd /home/eric-luo/Downloads/pointcloud-motion-planning/.claude/worktrees/rounded-box-base
PYTHONPATH=. /home/eric-luo/Downloads/pointcloud-motion-planning/.venv/bin/python /home/eric-luo/Downloads/pointcloud-motion-planning/scripts/gpu_capped.py --gpu-fraction 0.75 --gpu-util 0.7 \
  scripts/train_data_free_latent.py --output-dir "$OUT" --device cuda --resume --steps 9000 --set correction_cost_weight=50 --set correction_cost_band=0.05 \
  >> "$OUT/train.log" 2>&1
echo "finished (exit $?) $(date)" >> "$OUT/train.log"
