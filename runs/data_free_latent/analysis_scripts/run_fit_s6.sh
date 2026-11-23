#!/bin/bash
# Unseen-object fit of the no-base model S6 at 5,000 (same 225 objects, search settings and scoring as J4 and J7).
M=/home/eric-luo/Downloads/pointcloud-motion-planning
T=/home/eric-luo/.claude/jobs/8dde3c62/tmp
R=$M/runs/data_free_latent
D=$R/analysis_scripts/followup
CK=$R/S6_S5_continued_to_5k/checkpoint_last.pt
cd $M/.claude/worktrees/data-free-connectivity-plan || exit 1
PYTHONPATH=. $M/.venv/bin/python $M/scripts/gpu_capped.py --gpu-fraction 0.22 --gpu-util 0.7 \
  $D/heldout_fit.py $CK $M/datasets/data_free_anchors_20261112_v3/eligible.pt fit_S6_5000 500 8 4 > $D/fit_S6_5000.log 2>&1
PYTHONPATH=. $M/.venv/bin/python $M/scripts/gpu_capped.py --gpu-fraction 0.2 --gpu-util 0.7 \
  $T/rescore_fits.py $CK fit_S6_5000 $M/figures/data_free_worst_unseen_by_surface_S6_5000.png > $T/rescore_S6.out 2>&1
echo "done $(date)"
