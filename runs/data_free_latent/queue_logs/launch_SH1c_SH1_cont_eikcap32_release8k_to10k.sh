#!/usr/bin/env bash
# SH1c: SH1 (J4 recipe on 2,048 shared shells) continued 5,000 -> 10,000 with the slope (eikonal) multiplier capped at
# 32 (J4's late-training mean; SH1 ran 45) until 8,000, the cap raised geometrically to 100 over 8,000 -> 9,500, then
# no cap for the last 500 updates (owner's request: loosen the field constraint to let variety grow, then restore
# accuracy). Learning rate is constant in this recipe, so the continuation is seamless. Compare with SH1 at 5,000 and
# J4 at 5,000 on variety, concavity, topology and true-SDF accuracy (sdf_truth.py). Branch rounded-box-base, commit 48383fe.
set -uo pipefail
ROOT=/home/eric-luo/Downloads/pointcloud-motion-planning
OUT=/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/SH1c_SH1_cont_eikcap32_release8k_to10k
cd $ROOT
until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 60; done
python3 runs/data_free_latent/analysis_scripts/scan_cache.py --evict .venv/lib/python3.12/site-packages \
  $HOME/.local/share/uv/python .claude/worktrees/rounded-box-base/arm_planner .claude/worktrees/rounded-box-base/scripts \
  scripts /home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/SH1_J4_scratch_shells2048_shared_volw_5k/checkpoint_last.pt >> "$OUT/train.log" 2>&1 || true
[ -f "$OUT/checkpoint_last.pt" ] || { cp /home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/SH1_J4_scratch_shells2048_shared_volw_5k/checkpoint_last.pt "$OUT/checkpoint_start.pt"; cp "$OUT/checkpoint_start.pt" "$OUT/checkpoint_last.pt"; }
ln -sfn "$OUT/tb" $ROOT/runs/tensorboard_active/data_free_SH1c_SH1_cont_eikcap32_release8k_to10k
echo "started $(date) from SH1 at 5,000" >> "$OUT/train.log"
( for s in 6000 8000 9500; do
    until [ "$(tail -1 "$OUT/metrics.jsonl" 2>/dev/null | python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["step"])' 2>/dev/null || echo 0)" -ge $s ]; do sleep 60; done
    sleep 120; cp "$OUT/checkpoint_last.pt" "$OUT/checkpoint_$s.pt"
  done ) &
SNAPPER=$!
cd $ROOT/.claude/worktrees/rounded-box-base
PYTHONPATH=. $ROOT/.venv/bin/python $ROOT/scripts/gpu_capped.py --gpu-fraction 0.75 --gpu-util 0.7 \
  scripts/train_data_free_latent.py --output-dir "$OUT" --device cuda --resume --steps 10000 \
  --set field_cap=32 --set field_cap_until=8000 --set field_cap_release_steps=1500 --set field_cap_release_to=100 \
  >> "$OUT/train.log" 2>&1
echo "finished (exit $?) $(date)" >> "$OUT/train.log"
kill $SNAPPER 2>/dev/null || true
