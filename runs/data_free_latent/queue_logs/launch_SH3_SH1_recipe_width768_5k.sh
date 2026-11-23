#!/usr/bin/env bash
# SH3: SH1's recipe (J4 + 2,048 shared shells, pose/volume integrals volume-weighted; from scratch, seed 0, 5,000
# updates) with the main network widened 512 -> 768 (side network unchanged). To fit the 0.75 memory cap each update
# is 8 passes of 16 codes instead of 4 of 32 (same 128 codes; measured harmless at width 512: 1.96 vs 1.89 s), and
# anchors per pass 6 -> 3 (same share). Owner's request: does a larger main network ease the slope (eikonal)
# constraint and leave room for variety? Compare with SH1 and J4 at 1,000 / 2,500 / 5,000 on variety, concavity,
# topology, true-SDF accuracy, the eikonal multiplier, and rank / dead units. Branch rounded-box-base, commit 48383fe.
set -uo pipefail
ROOT=/home/eric-luo/Downloads/pointcloud-motion-planning
OUT=/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/SH3_SH1_recipe_width768_5k
SH1=/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/SH1_J4_scratch_shells2048_shared_volw_5k
cd $ROOT
until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 60; done
python3 runs/data_free_latent/analysis_scripts/scan_cache.py --evict .venv/lib/python3.12/site-packages \
  $HOME/.local/share/uv/python .claude/worktrees/rounded-box-base/arm_planner .claude/worktrees/rounded-box-base/scripts \
  scripts "$SH1/config.json" >> "$OUT/train.log" 2>&1 || true
ln -sfn "$OUT/tb" $ROOT/runs/tensorboard_active/data_free_SH3_SH1_recipe_width768_5k
echo "started $(date)" >> "$OUT/train.log"
( for s in 1000 2500; do
    until [ "$(tail -1 "$OUT/metrics.jsonl" 2>/dev/null | python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["step"])' 2>/dev/null || echo 0)" -ge $s ]; do sleep 60; done
    sleep 120; cp "$OUT/checkpoint_last.pt" "$OUT/checkpoint_$s.pt"
  done ) &
SNAPPER=$!
cd $ROOT/.claude/worktrees/rounded-box-base
PYTHONPATH=. $ROOT/.venv/bin/python $ROOT/scripts/gpu_capped.py --gpu-fraction 0.75 --gpu-util 0.7 \
  scripts/train_data_free_latent.py --output-dir "$OUT" --device cuda --config-from "$SH1/config.json" --steps 5000 --seed 0 \
  --set width=768 --set accumulate=8 --set codes_per_cluster=4 --set anchor_codes_per_pass=3 \
  --set diag_every=250 --set ckpt_every=100 --set log_every=25 --set diag_resolution=96 \
  >> "$OUT/train.log" 2>&1
echo "finished (exit $?) $(date)" >> "$OUT/train.log"
kill $SNAPPER 2>/dev/null || true
