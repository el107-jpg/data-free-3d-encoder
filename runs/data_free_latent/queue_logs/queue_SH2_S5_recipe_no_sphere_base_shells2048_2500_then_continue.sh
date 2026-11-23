#!/usr/bin/env bash
# Queued after SH1. SH2: the other session's S5 recipe (J4 recipe without the fixed sphere in the formula,
# main network started from the fitted sphere S0, outer limit by smooth minimum, anchor pull held off 500 then
# ramped over 500, slope term at all points; seed 0, 2,500 updates) with only the sample points changed:
# point_sampling=shells + pool_points=2047 (as SH1; pose/volume integrals volume-weighted). Compare with S5 at
# 500 / 1,000 / 2,000 / 2,500. Then SH2c: SH2 resumed to 5,000 exactly as S6 continued S5 (the cosine schedule
# restarts over 5,000); compare with S6 at 3,000 / 4,000 / 5,000. Owner's question: do the shells improve
# topology and stability enough to drop the sphere residual? Branch rounded-box-base, commit 5c1c1c2.
set -uo pipefail
ROOT=/home/eric-luo/Downloads/pointcloud-motion-planning
OUT=/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/SH2_S5_recipe_no_sphere_base_shells2048_2500
OUT2=/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/SH2c_SH2_continued_to_5k
S5=/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/S5_J4_recipe_sphere_start_no_base_anchor_hold_surface_slope_2500
SH1LOG=/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/SH1_J4_scratch_shells2048_shared_volw_5k/train.log
mkdir -p "$OUT" "$OUT2"
echo "queued $(date), waiting for SH1" >> "$OUT/train.log"
until grep -q "^finished" "$SH1LOG" 2>/dev/null; do sleep 60; done
sleep 30
snap() {  # snap <dir> <steps...>: copy checkpoint_last at each listed update
  local dir=$1; shift
  for s in "$@"; do
    until [ -f "$dir/metrics.jsonl" ] && [ "$(tail -1 "$dir/metrics.jsonl" | python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["step"])' 2>/dev/null || echo 0)" -ge $s ]; do sleep 60; done
    sleep 120; cp "$dir/checkpoint_last.pt" "$dir/checkpoint_$s.pt"
  done
}
cd $ROOT
python3 runs/data_free_latent/analysis_scripts/scan_cache.py --evict .venv/lib/python3.12/site-packages \
  $HOME/.local/share/uv/python .claude/worktrees/rounded-box-base/arm_planner .claude/worktrees/rounded-box-base/scripts \
  scripts "$S5/config.json" runs/data_free_latent/S0_sphere_fit_20261114/fourier3 >> "$OUT/train.log" 2>&1 || true
ln -sfn "$OUT/tb" $ROOT/runs/tensorboard_active/data_free_SH2_S5_recipe_no_sphere_base_shells2048_2500
echo "started $(date)" >> "$OUT/train.log"
( snap "$OUT" 500 1000 2000 ) &
SNAPPER=$!
cd $ROOT/.claude/worktrees/rounded-box-base
PYTHONPATH=. $ROOT/.venv/bin/python $ROOT/scripts/gpu_capped.py --gpu-fraction 0.75 --gpu-util 0.7 \
  scripts/train_data_free_latent.py --output-dir "$OUT" --device cuda --config-from "$S5/config.json" --steps 2500 --seed 0 \
  --set point_sampling=shells --set pool_points=2047 \
  --set diag_every=250 --set ckpt_every=100 --set log_every=25 --set diag_resolution=96 \
  >> "$OUT/train.log" 2>&1
CODE=$?
echo "finished (exit $CODE) $(date)" >> "$OUT/train.log"
kill $SNAPPER 2>/dev/null || true
[ $CODE -eq 0 ] || exit $CODE
cp "$OUT/checkpoint_last.pt" "$OUT/checkpoint_2500.pt"
# SH2c: continue to 5,000 in its own folder, as S6 did
cp "$OUT/checkpoint_last.pt" "$OUT2/checkpoint_start.pt"
cp "$OUT/checkpoint_last.pt" "$OUT2/checkpoint_last.pt"
cp "$OUT/config.json" "$OUT2/config.json" 2>/dev/null || true
cd $ROOT
python3 runs/data_free_latent/analysis_scripts/scan_cache.py --evict .venv/lib/python3.12/site-packages \
  $HOME/.local/share/uv/python .claude/worktrees/rounded-box-base/arm_planner .claude/worktrees/rounded-box-base/scripts \
  scripts >> "$OUT2/train.log" 2>&1 || true
ln -sfn "$OUT2/tb" $ROOT/runs/tensorboard_active/data_free_SH2c_SH2_continued_to_5k
echo "started $(date) from SH2 at 2,500" >> "$OUT2/train.log"
( snap "$OUT2" 3000 4000 ) &
SNAPPER=$!
cd $ROOT/.claude/worktrees/rounded-box-base
PYTHONPATH=. $ROOT/.venv/bin/python $ROOT/scripts/gpu_capped.py --gpu-fraction 0.75 --gpu-util 0.7 \
  scripts/train_data_free_latent.py --output-dir "$OUT2" --device cuda --resume --steps 5000 >> "$OUT2/train.log" 2>&1
echo "finished (exit $?) $(date)" >> "$OUT2/train.log"
kill $SNAPPER 2>/dev/null || true
