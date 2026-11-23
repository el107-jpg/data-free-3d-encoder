#!/usr/bin/env bash
# Base-variation regularisers, both continuing B6 from update 1,000 to 3,000 with B7's budget (0.5 by 2,500).
# Control: B7 at 2,000 and 3,000 (same start, no regulariser). Branch rounded-box-base, commit c7fb386.
#   B8: slope and consistency gradients into the base scaled by 0.1.
#   B9: adaptive spread constraint: std across codes of each extent >= 0.10, of the radius >= 0.05.
set -uo pipefail
cd /home/eric-luo/Downloads/pointcloud-motion-planning
while pgrep -f "train_data_free_latent[.]py --output-dir" > /dev/null; do sleep 30; done
python3 runs/data_free_latent/analysis_scripts/scan_cache.py --evict .venv/lib/python3.12/site-packages \
  $HOME/.local/share/uv/python .claude/worktrees/rounded-box-base/arm_planner .claude/worktrees/rounded-box-base/scripts \
  scripts > /home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/queue_logs/queue_B8_then_B9.scan.log 2>&1 || true
run() {
  local NAME=$1; shift
  local OUT=/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/$NAME
  ln -sfn "$OUT/tb" /home/eric-luo/Downloads/pointcloud-motion-planning/runs/tensorboard_active/data_free_$NAME
  echo "started $(date)" >> "$OUT/train.log"
  ( until [ -f "$OUT/metrics.jsonl" ] && [ "$(tail -1 "$OUT/metrics.jsonl" | python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["step"])' 2>/dev/null || echo 0)" -ge 2000 ]; do sleep 60; done
    cp "$OUT/checkpoint_last.pt" "$OUT/checkpoint_2000.pt" ) &
  ( cd /home/eric-luo/Downloads/pointcloud-motion-planning/.claude/worktrees/rounded-box-base && PYTHONPATH=. /home/eric-luo/Downloads/pointcloud-motion-planning/.venv/bin/python /home/eric-luo/Downloads/pointcloud-motion-planning/scripts/gpu_capped.py --gpu-fraction 0.75 --gpu-util 0.7 \
      scripts/train_data_free_latent.py --output-dir "$OUT" --device cuda --resume --steps 3000 \
      --set base_budget=0.5 --set base_budget_steps=2500 "$@" >> "$OUT/train.log" 2>&1 )
  echo "finished $(date)" >> "$OUT/train.log"
}
run B8_B6_cont_fieldgrad01_to3k --set base_eikonal_grad_scale=0.1 --set base_consistency_grad_scale=0.1
run B9_B6_cont_basespread_to3k --set base_spread_target=0.1 --set base_spread_target_r=0.05
