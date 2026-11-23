#!/usr/bin/env bash
# SH3c: SH3 (width 768, shells) continued 5,000 -> 10,000 with the variety-weight controller (1 to 4; grows 0.1% per
# update while smoothed consistency < 0.025 and bridge/delete/fill/widen are within tolerance, else shrinks 0.2%),
# slope (eikonal) ceiling 0.010 -> 0.015 and multiplier limit 200 -> 1000. Owner's request. rounded-box-base 52c2223.
set -uo pipefail
ROOT=/home/eric-luo/Downloads/pointcloud-motion-planning
OUT=/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/SH3c_SH3_cont_variety_controller_to10k
cd $ROOT
echo "queued $(date), waiting for the GPU" >> "$OUT/train.log"
free=0
while [ $free -lt 3 ]; do
  if apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader) && [ -z "$apps" ]; then free=$((free + 1)); else free=0; fi
  [ $free -lt 3 ] && sleep 60
done
python3 runs/data_free_latent/analysis_scripts/scan_cache.py --evict .venv/lib/python3.12/site-packages \
  $HOME/.local/share/uv/python .claude/worktrees/rounded-box-base/arm_planner .claude/worktrees/rounded-box-base/scripts \
  scripts /home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/SH3_SH1_recipe_width768_5k/checkpoint_last.pt >> "$OUT/train.log" 2>&1 || true
[ -f "$OUT/checkpoint_last.pt" ] || cp /home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/SH3_SH1_recipe_width768_5k/checkpoint_last.pt "$OUT/checkpoint_last.pt"
cp /home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/SH3_SH1_recipe_width768_5k/config.json "$OUT/config_SH3_start.json"
ln -sfn "$OUT/tb" $ROOT/runs/tensorboard_active/data_free_SH3c_SH3_cont_variety_controller_to10k
echo "started $(date)" >> "$OUT/train.log"
( for s in 6000 7500 9000; do
    until [ "$(tail -n 1 "$OUT/metrics.jsonl" 2>/dev/null | python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["step"])' 2>/dev/null || echo 0)" -ge $s ]; do sleep 60; done
    sleep 120; cp "$OUT/checkpoint_last.pt" "$OUT/checkpoint_$s.pt"
  done ) &
SNAPPER=$!
cd $ROOT/.claude/worktrees/rounded-box-base
PYTHONPATH=. $ROOT/.venv/bin/python $ROOT/scripts/gpu_capped.py --gpu-fraction 0.75 --gpu-util 0.7 \
  scripts/train_data_free_latent.py --output-dir "$OUT" --device cuda --resume --steps 10000 \
  --set variety_weight_max=4 --set variety_consistency_ceiling=0.025 --set tolerances.eikonal=0.015 --set mult_max=1000 \
  >> "$OUT/train.log" 2>&1
echo "finished (exit $?) $(date)" >> "$OUT/train.log"
kill $SNAPPER 2>/dev/null || true
