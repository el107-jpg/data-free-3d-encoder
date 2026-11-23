#!/bin/bash
# Evaluate S6 (S5 continued to 5,000) at 4,000 and 5,000 beside another training, with J4 at 5,000 as control.
M=/home/eric-luo/Downloads/pointcloud-motion-planning
T=/home/eric-luo/.claude/jobs/8dde3c62/tmp
F=$M/runs/data_free_latent/analysis_scripts/followup
R=$M/runs/data_free_latent
RUN=$R/S6_S5_continued_to_5k
J4=$R/J4_d64_nocentre_procedural96_even_code_leak05_beta100_fourier3_5k
cd $M/.claude/worktrees/data-free-connectivity-plan || exit 1
run() { PYTHONPATH=. $M/.venv/bin/python $M/scripts/gpu_capped.py --gpu-fraction 0.2 --gpu-util 0.4 "$@" 2>&1 | grep -v "^\[gpu_capped\]"; }
cat $T/S6_launch.out; date
grep -E "^diag" $RUN/train.log | cut -c1-215
ls $RUN | grep checkpoint | tr '\n' ' '; echo
echo "== 5000 evalrun"; run $F/evalrun.py $RUN/checkpoint_last.pt S6_5000 | tail -n 5
echo "== 4000 evalrun"; run $F/evalrun.py $RUN/checkpoint_4000.pt S6_4000 | grep -E "topology|anchors"
echo "== variance (S6 at 3000, 4000, 5000; J4 at 5000)"; run $T/variance_noise.py $RUN/checkpoint_3000.pt $RUN/checkpoint_4000.pt $RUN/checkpoint_last.pt $J4/checkpoint_last.pt
echo "== field quality (S6, J4 at 5000)"; run $T/field_quality.py $RUN/checkpoint_last.pt $J4/checkpoint_last.pt
echo "== facing (S6, J4 at 5000)"; run $T/facing.py $RUN/checkpoint_last.pt $J4/checkpoint_last.pt
echo "== sharpness"; run $T/sharpness.py $T/sharpness_S6.json "J4_5000=$J4/checkpoint_last.pt" "S6_5000=$RUN/checkpoint_last.pt"
echo "== dead units"; run $T/unit_activity.py $RUN/checkpoint_last.pt | grep -A10 "^S6" | head -12
echo "== field error over the run"
python3 $T/field_error_over_run.py S6=$RUN | head -3
echo "== sheets and contours"
run $T/decoded_sheet.py $RUN/checkpoint_last.pt S6_5000 | tail -n 2
run $T/field_contours3.py $RUN/checkpoint_last.pt $M/figures/data_free_field_sphere_start_S6_update5000.png "S6 (no fixed sphere; S5 continued)"
run $T/crease_pairs.py $M/figures/data_free_no_base_S6_vs_J4_5000.png "5,000 updates: sphere as a fixed base (top) and no base, fitted-sphere start with delayed anchors (bottom). Same codes, separate trainings; each panel zoomed to its shape." "J4 (sphere base)=$J4/checkpoint_last.pt" "S6 (no base)=$RUN/checkpoint_last.pt" codes=8
echo "== done $(date)"
