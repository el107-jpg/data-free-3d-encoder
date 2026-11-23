#!/bin/bash
# Evaluate the no-base continuation S7 at 6,000 / 7,000 / 8,000 / 9,000 / 10,000: validity, anchors, variance on
# 96 codes, pairwise variety, layer dimensions; unseen-object fit at 10,000. J6 at matching points as control.
M=/home/eric-luo/Downloads/pointcloud-motion-planning
T=/home/eric-luo/.claude/jobs/8dde3c62/tmp
F=$M/runs/data_free_latent/analysis_scripts/followup
R=$M/runs/data_free_latent
RUN=$R/S7_S6_continued_to_10k
J6=$R/J6_J4_continued_to_10k
cd $M/.claude/worktrees/data-free-connectivity-plan || exit 1
run() { PYTHONPATH=. $M/.venv/bin/python $M/scripts/gpu_capped.py --gpu-fraction 0.2 --gpu-util 0.5 "$@" 2>&1 | grep -v "^\[gpu_capped\]"; }
cat $T/S7_launch.out; date
grep -E "^diag" $RUN/train.log | cut -c1-215
for u in 6000 7000 8000 9000 10000; do
  echo "== $u"; run $F/evalrun.py $RUN/checkpoint_$u.pt S7_$u | grep -E "topology|anchors"
done
echo "== variance on 96 codes"; run $T/variance_noise.py $RUN/checkpoint_6000.pt $RUN/checkpoint_7000.pt $RUN/checkpoint_8000.pt $RUN/checkpoint_9000.pt $RUN/checkpoint_10000.pt $J6/checkpoint_last.pt
echo "== pairwise variety"; run $T/variety_pairs.py $T/variety_pairs_S7.json "S7 at 7,000=$RUN/checkpoint_7000.pt" "S7 at 10,000=$RUN/checkpoint_10000.pt" "J6 at 7,500=$J6/checkpoint_7500.pt" "J6 at 10,000=$J6/checkpoint_last.pt"
echo "== layer dimensions"; run $T/layer_dims.py $T/layer_dims_S7.json "no base 6000=$RUN/checkpoint_6000.pt" "no base 7000=$RUN/checkpoint_7000.pt" "no base 8000=$RUN/checkpoint_8000.pt" "no base 9000=$RUN/checkpoint_9000.pt" "no base 10000=$RUN/checkpoint_10000.pt" "J6 10000=$J6/checkpoint_last.pt" | grep -E "^no base|^J6|main network, hidden layer 1 |scale and shift for main layer 1|surface motion"
echo "== field quality"; run $T/field_quality.py $RUN/checkpoint_10000.pt
echo "== sheets"; run $T/decoded_sheet.py $RUN/checkpoint_10000.pt S7_10000 | tail -n 2
echo "== unseen-object fit at 10,000"
PYTHONPATH=. $M/.venv/bin/python $M/scripts/gpu_capped.py --gpu-fraction 0.22 --gpu-util 0.7 \
  $F/heldout_fit.py $RUN/checkpoint_10000.pt $M/datasets/data_free_anchors_20261112_v3/eligible.pt fit_S7_10000 500 8 4 > $F/fit_S7_10000.log 2>&1
grep -v gpu_capped $F/fit_S7_10000.log | head -3 | cut -c1-250
run $T/rescore_fits.py $RUN/checkpoint_10000.pt fit_S7_10000 | head -6 | cut -c1-200
echo "== done $(date)"
