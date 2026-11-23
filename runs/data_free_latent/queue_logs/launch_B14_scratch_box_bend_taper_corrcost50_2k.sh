#!/usr/bin/env bash
# B14: from scratch on the J4 recipe with B12's base settings from update 0 (linear head on FiLM's features through
# tanh, budget 0.01 -> 0.5 over 2,500, spread push extents 0.10 / radius 0.05, base head 0.1x learning rate,
# correction cost 50) plus bends (6) and tapers (3) with soft-centroid recentring and spread targets
# bend 1.0 / taper 0.3. 2,000 updates. Branch rounded-box-base, commit 69c15a2.
set -uo pipefail
OUT=/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/B14_scratch_box_bend_taper_corrcost50_2k
J4=/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/J4_d64_nocentre_procedural96_even_code_leak05_beta100_fourier3_5k
cd /home/eric-luo/Downloads/pointcloud-motion-planning
while pgrep -f "train_data_free_latent[.]py --output-dir" > /dev/null; do sleep 30; done
python3 runs/data_free_latent/analysis_scripts/scan_cache.py --evict .venv/lib/python3.12/site-packages \
  $HOME/.local/share/uv/python .claude/worktrees/rounded-box-base/arm_planner .claude/worktrees/rounded-box-base/scripts \
  scripts "$J4/config.json" >> "$OUT/train.log" 2>&1 || true
ln -sfn "$OUT/tb" /home/eric-luo/Downloads/pointcloud-motion-planning/runs/tensorboard_active/data_free_B14_scratch_box_bend_taper_corrcost50_2k
echo "started $(date)" >> "$OUT/train.log"
( until [ -f "$OUT/metrics.jsonl" ] && [ "$(tail -1 "$OUT/metrics.jsonl" | python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["step"])' 2>/dev/null || echo 0)" -ge 1000 ]; do sleep 60; done
  cp "$OUT/checkpoint_last.pt" "$OUT/checkpoint_1000.pt" ) &
SNAPPER=$!
cd /home/eric-luo/Downloads/pointcloud-motion-planning/.claude/worktrees/rounded-box-base
PYTHONPATH=. /home/eric-luo/Downloads/pointcloud-motion-planning/.venv/bin/python /home/eric-luo/Downloads/pointcloud-motion-planning/scripts/gpu_capped.py --gpu-fraction 0.75 --gpu-util 0.7 \
  scripts/train_data_free_latent.py --output-dir "$OUT" --device cuda --config-from "$J4/config.json" --steps 2000 --seed 0 \
  --set base_shape=rounded_box --set base_head_form=linear --set base_head_net=shared_mlp \
  --set base_budget=0.5 --set base_budget_start=0.01 --set base_budget_steps=2500 \
  --set base_spread_target=0.1 --set base_spread_target_r=0.05 --set base_lr_scale=0.1 \
  --set correction_cost_weight=50 --set correction_cost_band=0.05 \
  --set base_deform=True --set base_spread_target_bend=1.0 --set base_spread_target_taper=0.3 \
  --set diag_every=250 --set ckpt_every=100 --set log_every=25 --set diag_resolution=96 \
  >> "$OUT/train.log" 2>&1
echo "finished (exit $?) $(date)" >> "$OUT/train.log"
kill $SNAPPER 2>/dev/null || true
