#!/bin/bash
# Evaluate the distilled no-base model: right after distillation (5,000) and at 6,000 to 10,000, against J4 at 5,000
# and J6 (J4 continued over the same stretch).
M=/home/eric-luo/Downloads/pointcloud-motion-planning
T=/home/eric-luo/.claude/jobs/8dde3c62/tmp
F=$M/runs/data_free_latent/analysis_scripts/followup
R=$M/runs/data_free_latent
RUN=$R/S8_distilled_no_base_continued_to_10k
D0=$R/D0_distill_J4_5000_no_base/distilled_5000.pt
J4=$R/J4_d64_nocentre_procedural96_even_code_leak05_beta100_fourier3_5k
J6=$R/J6_J4_continued_to_10k
FR=${FRAC:-0.75}
cd $M/.claude/worktrees/data-free-connectivity-plan || exit 1
python3 $R/analysis_scripts/scan_cache.py --evict $M/.venv/lib/python3.12/site-packages /home/eric-luo/.local/share/uv/python \
  $M/.claude/worktrees/data-free-connectivity-plan/arm_planner $RUN $R/D0_distill_J4_5000_no_base $M/datasets/data_free_anchors_20261112_v3 2>&1 | tail -1
run() { PYTHONPATH=. $M/.venv/bin/python $M/scripts/gpu_capped.py --gpu-fraction $FR --gpu-util 0.7 "$@" 2>&1 | grep -v "^\[gpu_capped\]"; }
date; grep -E "^diag" $RUN/train.log | cut -c1-215
grep -E "^before|^after" $R/D0_distill_J4_5000_no_base/distill.log
echo "== distilled (5,000)"; run $F/evalrun.py $D0 S8_5000 | grep -E "topology|anchors"
for u in 6000 7000 8000 9000 10000; do
  echo "== $u"; run $F/evalrun.py $RUN/checkpoint_$u.pt S8_$u | grep -E "topology|anchors"
done
echo "== variance on 96 codes"; run $T/variance_noise.py $D0 $RUN/checkpoint_6000.pt $RUN/checkpoint_7000.pt $RUN/checkpoint_8000.pt $RUN/checkpoint_9000.pt $RUN/checkpoint_10000.pt
echo "== J6 variance at 6,000 and 7,500 already measured (1.09, 1.06); 9,000 0.84; 10,000 1.42"
echo "== layer injection"; run $T/layer_inject.py $T/layer_inject_S8.json "distilled 5000=$D0" "S8 7000=$RUN/checkpoint_7000.pt" "S8 10000=$RUN/checkpoint_10000.pt" "J6 10000=$J6/checkpoint_last.pt"
echo "== pairwise variety"; run $T/variety_pairs.py $T/variety_pairs_S8.json "S8 at 10,000=$RUN/checkpoint_10000.pt"
echo "== facing"; run $T/facing.py $D0 $RUN/checkpoint_10000.pt
echo "== field quality"; run $T/field_quality.py $D0 $RUN/checkpoint_10000.pt
echo "== sheets"; run $T/decoded_sheet.py $RUN/checkpoint_10000.pt S8_10000 | tail -n 2
run $T/crease_pairs.py $M/figures/data_free_distilled_no_base_vs_J4.png "Same codes: J4 at 5,000 (top), its no-base copy right after distillation (middle), and that copy trained on to 10,000 (bottom); each panel zoomed to its shape." "J4 at 5,000=$J4/checkpoint_last.pt" "no-base copy at 5,000=$D0" "no-base copy at 10,000=$RUN/checkpoint_10000.pt" codes=8
echo "== unseen-object fit at 10,000"
PYTHONPATH=. $M/.venv/bin/python $M/scripts/gpu_capped.py --gpu-fraction $FR --gpu-util 0.7 \
  $F/heldout_fit.py $RUN/checkpoint_10000.pt $M/datasets/data_free_anchors_20261112_v3/eligible.pt fit_S8_10000 500 8 4 > $F/fit_S8_10000.log 2>&1
grep -v gpu_capped $F/fit_S8_10000.log | head -3 | cut -c1-250
run $T/rescore_fits.py $RUN/checkpoint_10000.pt fit_S8_10000 | head -6 | cut -c1-200
echo "== done $(date)"
