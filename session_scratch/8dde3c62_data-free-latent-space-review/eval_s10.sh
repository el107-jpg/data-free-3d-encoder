#!/bin/bash
# Evaluate S10 (S9 continued to 20,000 at the low rate): sphere-bias measures over the run, validity, variety,
# per-layer code use, unseen-object fit at 16,000 and 20,000. References: S9 at 12,000 and J7 at 11,000 / 12,000.
M=/home/eric-luo/Downloads/pointcloud-motion-planning
T=/home/eric-luo/.claude/jobs/8dde3c62/tmp
F=$M/runs/data_free_latent/analysis_scripts/followup
R=$M/runs/data_free_latent
RUN=$R/S10_S9_continued_to_20k
S9=$R/S9_S8_lower_lr_to_12k/checkpoint_12000.pt
J7=$R/J7_J6_lower_lr_to_12k
cd $M/.claude/worktrees/data-free-connectivity-plan || exit 1
python3 $R/analysis_scripts/scan_cache.py --evict $M/.venv/lib/python3.12/site-packages /home/eric-luo/.local/share/uv/python \
  $M/.claude/worktrees/data-free-connectivity-plan/arm_planner $RUN $M/datasets/data_free_anchors_20261112_v3 2>&1 | tail -1
if [ -z "$(pgrep -f 'train_data_free_latent.py --output-dir')" ]; then FR=0.6; else FR=0.2; fi
run() { PYTHONPATH=. $M/.venv/bin/python $M/scripts/gpu_capped.py --gpu-fraction $FR --gpu-util 0.7 "$@" 2>&1 | grep -v "^\[gpu_capped\]"; }
date; grep -E "^diag" $RUN/train.log | awk 'NR % 4 == 1' | cut -c1-215
for u in 14000 16000 18000 20000; do
  echo "== $u"; run $F/evalrun.py $RUN/checkpoint_$u.pt S10_$u | grep -E "topology|anchors"
done
echo "== variance on 96 codes"; run $T/variance_noise.py $RUN/checkpoint_14000.pt $RUN/checkpoint_16000.pt $RUN/checkpoint_18000.pt $RUN/checkpoint_20000.pt
echo "== facing (sphere-bias measure)"; run $T/facing.py $S9 $RUN/checkpoint_14000.pt $RUN/checkpoint_16000.pt $RUN/checkpoint_18000.pt $RUN/checkpoint_20000.pt $J7/checkpoint_last.pt
echo "== layer injection"; run $T/layer_inject.py $T/layer_inject_S10.json "S10 16000=$RUN/checkpoint_16000.pt" "S10 20000=$RUN/checkpoint_20000.pt"
echo "== pairwise variety"; run $T/variety_pairs.py $T/variety_pairs_S10.json "S10 at 16,000=$RUN/checkpoint_16000.pt" "S10 at 20,000=$RUN/checkpoint_20000.pt"
echo "== field quality"; run $T/field_quality.py $RUN/checkpoint_20000.pt
echo "== sheets"; run $T/decoded_sheet.py $RUN/checkpoint_20000.pt S10_20000 | tail -n 2
for u in 16000 20000; do
  echo "== unseen-object fit at $u"
  PYTHONPATH=. $M/.venv/bin/python $M/scripts/gpu_capped.py --gpu-fraction $FR --gpu-util 0.7 \
    $F/heldout_fit.py $RUN/checkpoint_$u.pt $M/datasets/data_free_anchors_20261112_v3/eligible.pt fit_S10_$u 500 8 4 > $F/fit_S10_$u.log 2>&1
  grep -v gpu_capped $F/fit_S10_$u.log | head -1 | cut -c1-250
  run $T/rescore_fits.py $RUN/checkpoint_$u.pt fit_S10_$u | head -6 | cut -c1-200
done
P=$M/.venv/bin/python
$P $T/paired_fit.py fit_S9_12000 fit_S10_16000; $P $T/paired_fit.py fit_S9_12000 fit_S10_20000; $P $T/paired_fit.py fit_J7_11000 fit_S10_20000
echo "== done $(date)"
