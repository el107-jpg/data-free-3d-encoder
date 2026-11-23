#!/usr/bin/env bash
# Owner's rule (15 Nov, ~01:15): if B3's base has not improved by update 1,000, run a from-scratch learned-base
# run (B5) before B4; otherwise run B4 as planned. "Improved" = at B3's last check, over 1,024 fresh codes:
# base/field_variance_share >= 0.05, or radius range (max - min) >= 0.05, or largest extent >= 0.05.
set -euo pipefail
B3=/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/B3_J4_5000_box_linear_head_budget02_1k
while kill -0 516907 2> /dev/null; do sleep 30; done
decision=$(python3 - "$B3/diagnostics.jsonl" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
d = rows[-1]
share = d.get("base/field_variance_share", 0.0)
r_range = d.get("base/r_max", 0.5) - d.get("base/r_min", 0.5)
extent = max(d.get("base/b1_max", 0.0), d.get("base/b2_max", 0.0), d.get("base/b3_max", 0.0))
improved = share >= 0.05 or r_range >= 0.05 or extent >= 0.05
print(f"{'B4' if improved else 'B5'} step={d['step']} share={share:.4f} r_range={r_range:.4f} extent={extent:.4f}")
PY
)
echo "$(date) decision: $decision"
case "$decision" in
  B4*) exec /home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/queue_logs/launch_B4_box_linear_sepnet_budget.sh ;;
esac
# ---- B5: from scratch, B3's base design (linear head on the side network, deviation budget 0.01 -> 0.2 over 500)
NAME=B5_scratch_box_linear_head_budget02_1k
OUT=/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/$NAME
J4=/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/J4_d64_nocentre_procedural96_even_code_leak05_beta100_fourier3_5k
while pgrep -f "train_data_free_latent.py --output-dir" > /dev/null; do sleep 30; done
mkdir -p "$OUT"
cd /home/eric-luo/Downloads/pointcloud-motion-planning
python3 runs/data_free_latent/analysis_scripts/scan_cache.py --evict .venv/lib/python3.12/site-packages \
  $HOME/.local/share/uv/python .claude/worktrees/rounded-box-base/arm_planner .claude/worktrees/rounded-box-base/scripts \
  scripts >> "$OUT/train.log" 2>&1 || true
ln -sfn "$OUT/tb" /home/eric-luo/Downloads/pointcloud-motion-planning/runs/tensorboard_active/data_free_$NAME
echo "started $(date) (B3 decision: $decision)" >> "$OUT/train.log"
cd /home/eric-luo/Downloads/pointcloud-motion-planning/.claude/worktrees/rounded-box-base
PYTHONPATH=. exec /home/eric-luo/Downloads/pointcloud-motion-planning/.venv/bin/python /home/eric-luo/Downloads/pointcloud-motion-planning/scripts/gpu_capped.py --gpu-fraction 0.75 --gpu-util 0.7 \
  scripts/train_data_free_latent.py --output-dir "$OUT" --device cuda \
  --config-from "$J4/config.json" --steps 1000 --seed 0 \
  --set base_shape=rounded_box --set base_head_form=linear \
  --set base_budget=0.2 --set base_budget_start=0.01 --set base_budget_steps=500 \
  --set diag_every=250 --set ckpt_every=100 --set log_every=25 --set diag_resolution=96 \
  >> "$OUT/train.log" 2>&1
