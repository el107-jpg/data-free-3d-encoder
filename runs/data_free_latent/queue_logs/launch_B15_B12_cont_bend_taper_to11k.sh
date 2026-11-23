#!/usr/bin/env bash
# B15: B12 (cost 50) resumed at 9,000 into the base with 6 bends + 3 tapers (new head rows start at zero: same field),
# soft-centroid recentring; spread push bounded (weight <= 100, tolerance 1e-3) with bend/taper targets
# (1.0 / 0.3) growing from 0 over 9,000-10,000. Everything else as B12. 2,000 updates.
# Branch rounded-box-base, commit 81d6118. Controls: B12 at 9,000; B13 (cost 100) and B12c are other branches.
set -uo pipefail
OUT=/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/B15_B12_cont_bend_taper_to11k
cd /home/eric-luo/Downloads/pointcloud-motion-planning
while pgrep -f "train_data_free_latent[.]py --output-dir" > /dev/null; do sleep 30; done
python3 runs/data_free_latent/analysis_scripts/scan_cache.py --evict .venv/lib/python3.12/site-packages \
  $HOME/.local/share/uv/python .claude/worktrees/rounded-box-base/arm_planner .claude/worktrees/rounded-box-base/scripts \
  scripts "$OUT" >> "$OUT/train.log" 2>&1 || true
ln -sfn "$OUT/tb" /home/eric-luo/Downloads/pointcloud-motion-planning/runs/tensorboard_active/data_free_B15_B12_cont_bend_taper_to11k
echo "started $(date)" >> "$OUT/train.log"
( until [ -f "$OUT/metrics.jsonl" ] && [ "$(tail -1 "$OUT/metrics.jsonl" | python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["step"])' 2>/dev/null || echo 0)" -ge 10000 ]; do sleep 60; done
  cp "$OUT/checkpoint_last.pt" "$OUT/checkpoint_10000.pt" ) &
SNAPPER=$!
cd /home/eric-luo/Downloads/pointcloud-motion-planning/.claude/worktrees/rounded-box-base
PYTHONPATH=. /home/eric-luo/Downloads/pointcloud-motion-planning/.venv/bin/python /home/eric-luo/Downloads/pointcloud-motion-planning/scripts/gpu_capped.py --gpu-fraction 0.75 --gpu-util 0.7 \
  scripts/train_data_free_latent.py --output-dir "$OUT" --device cuda --resume --steps 11000 \
  --set base_deform=True --set base_spread_target_bend=1.0 --set base_spread_target_taper=0.3 \
  --set base_spread_mult_max=100 --set base_spread_tolerance=1e-3 --set base_spread_ramp_from=9000 --set base_spread_ramp_steps=1000 \
  >> "$OUT/train.log" 2>&1
echo "finished (exit $?) $(date)" >> "$OUT/train.log"
kill $SNAPPER 2>/dev/null || true
