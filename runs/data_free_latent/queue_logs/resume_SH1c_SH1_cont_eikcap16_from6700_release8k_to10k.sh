#!/usr/bin/env bash
# SH1c resumed at 6,700 with the slope (eikonal) multiplier capped at 16 (the cap of 32 used for 5,000 -> 6,700 never
# bound: the multiplier sat at 30-32 on its own). Cap 16 until 8,000, raised geometrically to 100 over 8,000 -> 9,500,
# then uncapped to 10,000. Owner's request. Branch rounded-box-base, commit 48383fe.
set -uo pipefail
ROOT=/home/eric-luo/Downloads/pointcloud-motion-planning
OUT=/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/SH1c_SH1_cont_eikcap16_from6700_release8k_to10k
cd $ROOT
python3 runs/data_free_latent/analysis_scripts/scan_cache.py --evict .venv/lib/python3.12/site-packages \
  $HOME/.local/share/uv/python .claude/worktrees/rounded-box-base/arm_planner .claude/worktrees/rounded-box-base/scripts \
  scripts "$OUT/checkpoint_last.pt" >> "$OUT/train.log" 2>&1 || true
ln -sfn "$OUT/tb" $ROOT/runs/tensorboard_active/data_free_SH1c_SH1_cont_eikcap16_from6700_release8k_to10k
( for s in 8000 9500; do
    until [ "$(tail -1 "$OUT/metrics.jsonl" 2>/dev/null | python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["step"])' 2>/dev/null || echo 0)" -ge $s ]; do sleep 60; done
    sleep 120; cp "$OUT/checkpoint_last.pt" "$OUT/checkpoint_$s.pt"
  done ) &
SNAPPER=$!
cd $ROOT/.claude/worktrees/rounded-box-base
PYTHONPATH=. $ROOT/.venv/bin/python $ROOT/scripts/gpu_capped.py --gpu-fraction 0.75 --gpu-util 0.7 \
  scripts/train_data_free_latent.py --output-dir "$OUT" --device cuda --resume --steps 10000 \
  --set field_cap=16 --set field_cap_until=8000 --set field_cap_release_steps=1500 --set field_cap_release_to=100 \
  >> "$OUT/train.log" 2>&1
echo "finished (exit $?) $(date)" >> "$OUT/train.log"
kill $SNAPPER 2>/dev/null || true
