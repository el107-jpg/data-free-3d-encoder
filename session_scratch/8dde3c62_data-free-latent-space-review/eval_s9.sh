#!/bin/bash
# Evaluate S9 at 10,500 / 11,000 / 11,500 / 12,000 against J7 (same stretch with the base), plus the unseen-object
# fit at 12,000 and a redraw of the unseen-object comparison pictures.
M=/home/eric-luo/Downloads/pointcloud-motion-planning
T=/home/eric-luo/.claude/jobs/8dde3c62/tmp
F=$M/runs/data_free_latent/analysis_scripts/followup
R=$M/runs/data_free_latent
RUN=$R/S9_S8_lower_lr_to_12k
J7=$R/J7_J6_lower_lr_to_12k
cd $M/.claude/worktrees/data-free-connectivity-plan || exit 1
python3 $R/analysis_scripts/scan_cache.py --evict $M/.venv/lib/python3.12/site-packages /home/eric-luo/.local/share/uv/python \
  $M/.claude/worktrees/data-free-connectivity-plan/arm_planner $RUN $M/datasets/data_free_anchors_20261112_v3 2>&1 | tail -1
if [ -z "$(pgrep -f 'train_data_free_latent.py --output-dir')" ]; then FR=0.6; else FR=0.2; fi
run() { PYTHONPATH=. $M/.venv/bin/python $M/scripts/gpu_capped.py --gpu-fraction $FR --gpu-util 0.7 "$@" 2>&1 | grep -v "^\[gpu_capped\]"; }
date; grep -E "^diag" $RUN/train.log | cut -c1-215
for u in 10500 11000 11500 12000; do
  echo "== $u"; run $F/evalrun.py $RUN/checkpoint_$u.pt S9_$u | grep -E "topology|anchors"
done
echo "== variance on 96 codes"; run $T/variance_noise.py $RUN/checkpoint_11000.pt $RUN/checkpoint_12000.pt $J7/checkpoint_11000.pt $J7/checkpoint_last.pt
echo "== layer injection"; run $T/layer_inject.py $T/layer_inject_S9.json "S9 12000=$RUN/checkpoint_12000.pt" "J7 12000=$J7/checkpoint_last.pt"
echo "== pairwise variety"; run $T/variety_pairs.py $T/variety_pairs_S9.json "S9 at 12,000=$RUN/checkpoint_12000.pt" "J7 at 11,000=$J7/checkpoint_11000.pt"
echo "== facing"; run $T/facing.py $RUN/checkpoint_12000.pt $J7/checkpoint_11000.pt
echo "== field quality"; run $T/field_quality.py $RUN/checkpoint_12000.pt $J7/checkpoint_11000.pt
echo "== sheets"; run $T/decoded_sheet.py $RUN/checkpoint_12000.pt S9_12000 | tail -n 2
echo "== unseen-object fit at 12,000"
PYTHONPATH=. $M/.venv/bin/python $M/scripts/gpu_capped.py --gpu-fraction $FR --gpu-util 0.7 \
  $F/heldout_fit.py $RUN/checkpoint_12000.pt $M/datasets/data_free_anchors_20261112_v3/eligible.pt fit_S9_12000 500 8 4 > $F/fit_S9_12000.log 2>&1
grep -v gpu_capped $F/fit_S9_12000.log | head -3 | cut -c1-250
run $T/rescore_fits.py $RUN/checkpoint_12000.pt fit_S9_12000 | head -6 | cut -c1-200
echo "== done $(date)"
