#!/usr/bin/env bash
# SH1: J4 from scratch (seed 0, 5,000 updates), identical except the sample points: point_sampling=shells
# (5/50/20/25 % of points at radii 0-0.2/0.2-0.55/0.55-0.7/0.7-0.9, counted equally) and pool_points=2047, so all
# 2047 non-origin points are shared by the update's 4 passes and the diversity distance uses all of them
# (J4: uniform box points, 1,024 shared). Every term uses the shells. Pose and volume estimates (centroid, size, axes, min volume) are volume-weighted so they match uniform sampling. Branch rounded-box-base, commit 5c1c1c2. (S1x_aborted_shells_biased_pose_500 was the first try without that weighting.)
# Compare with J4 at 1,000 / 2,500 / 5,000.
set -uo pipefail
OUT=/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/SH1_J4_scratch_shells2048_shared_volw_5k
J4=/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/J4_d64_nocentre_procedural96_even_code_leak05_beta100_fourier3_5k
cd /home/eric-luo/Downloads/pointcloud-motion-planning
while pgrep -f "train_data_free_latent[.]py --output-dir" > /dev/null; do sleep 30; done
python3 runs/data_free_latent/analysis_scripts/scan_cache.py --evict .venv/lib/python3.12/site-packages \
  $HOME/.local/share/uv/python .claude/worktrees/rounded-box-base/arm_planner .claude/worktrees/rounded-box-base/scripts \
  scripts "$J4/config.json" >> "$OUT/train.log" 2>&1 || true
ln -sfn "$OUT/tb" /home/eric-luo/Downloads/pointcloud-motion-planning/runs/tensorboard_active/data_free_SH1_J4_scratch_shells2048_shared_volw_5k
echo "started $(date)" >> "$OUT/train.log"
( for s in 1000 2500; do
    until [ -f "$OUT/metrics.jsonl" ] && [ "$(tail -1 "$OUT/metrics.jsonl" | python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["step"])' 2>/dev/null || echo 0)" -ge $s ]; do sleep 60; done
    sleep 120; cp "$OUT/checkpoint_last.pt" "$OUT/checkpoint_$s.pt"
  done ) &
SNAPPER=$!
cd /home/eric-luo/Downloads/pointcloud-motion-planning/.claude/worktrees/rounded-box-base
PYTHONPATH=. /home/eric-luo/Downloads/pointcloud-motion-planning/.venv/bin/python /home/eric-luo/Downloads/pointcloud-motion-planning/scripts/gpu_capped.py --gpu-fraction 0.75 --gpu-util 0.7 \
  scripts/train_data_free_latent.py --output-dir "$OUT" --device cuda --config-from "$J4/config.json" --steps 5000 --seed 0 \
  --set point_sampling=shells --set pool_points=2047 \
  --set diag_every=250 --set ckpt_every=100 --set log_every=25 --set diag_resolution=96 \
  >> "$OUT/train.log" 2>&1
echo "finished (exit $?) $(date)" >> "$OUT/train.log"
kill $SNAPPER 2>/dev/null || true
