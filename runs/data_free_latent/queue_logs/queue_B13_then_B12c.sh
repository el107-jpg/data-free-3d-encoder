#!/usr/bin/env bash
# B13: B12 continued 9,000 -> 11,000 with the correction-size cost raised to 100.
# B12c: control for B12: B11 continued 7,000 -> 9,000 with the cost held at 25 (B12 used 50).
# Everything else as B10-B12 (base head 0.1x lr, spread push, budget). Branch rounded-box-base, commit 2b033aa.
set -uo pipefail
cd /home/eric-luo/Downloads/pointcloud-motion-planning
run() {
  local NAME=$1 STEPS=$2 SNAP=$3; shift 3
  local OUT=/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/$NAME
  while pgrep -f "train_data_free_latent[.]py --output-dir" > /dev/null; do sleep 30; done
  python3 runs/data_free_latent/analysis_scripts/scan_cache.py --evict .venv/lib/python3.12/site-packages \
    $HOME/.local/share/uv/python .claude/worktrees/rounded-box-base/arm_planner .claude/worktrees/rounded-box-base/scripts \
    scripts "$OUT" >> "$OUT/train.log" 2>&1 || true
  ln -sfn "$OUT/tb" /home/eric-luo/Downloads/pointcloud-motion-planning/runs/tensorboard_active/data_free_$NAME
  echo "started $(date)" >> "$OUT/train.log"
  ( until [ -f "$OUT/metrics.jsonl" ] && [ "$(tail -1 "$OUT/metrics.jsonl" | python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["step"])' 2>/dev/null || echo 0)" -ge $SNAP ]; do sleep 60; done
    cp "$OUT/checkpoint_last.pt" "$OUT/checkpoint_$SNAP.pt" ) &
  local SNAPPER=$!
  ( cd /home/eric-luo/Downloads/pointcloud-motion-planning/.claude/worktrees/rounded-box-base && PYTHONPATH=. /home/eric-luo/Downloads/pointcloud-motion-planning/.venv/bin/python /home/eric-luo/Downloads/pointcloud-motion-planning/scripts/gpu_capped.py --gpu-fraction 0.75 --gpu-util 0.7 \
      scripts/train_data_free_latent.py --output-dir "$OUT" --device cuda --resume --steps $STEPS "$@" >> "$OUT/train.log" 2>&1 )
  echo "finished (exit $?) $(date)" >> "$OUT/train.log"
  kill $SNAPPER 2>/dev/null || true
}
run B13_B12_cont_corrcost100_to11k 11000 10000 --set correction_cost_weight=100
run B12c_B11_cont_corrcost25_to9k_control 9000 8000 --set correction_cost_weight=25
