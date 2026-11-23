#!/bin/bash
# Rerun the unseen-object fit of S7 at 10,000 (the grid file was found damaged in memory afterwards), then score it.
M=/home/eric-luo/Downloads/pointcloud-motion-planning
T=/home/eric-luo/.claude/jobs/8dde3c62/tmp
R=$M/runs/data_free_latent
D=$R/analysis_scripts/followup
CK=$R/S7_S6_continued_to_10k/checkpoint_10000.pt
cd $M/.claude/worktrees/data-free-connectivity-plan || exit 1
python3 $R/analysis_scripts/scan_cache.py --evict $M/.venv/lib/python3.12/site-packages /home/eric-luo/.local/share/uv/python \
  $M/.claude/worktrees/data-free-connectivity-plan/arm_planner $M/datasets/data_free_anchors_20261112_v3 $R/S7_S6_continued_to_10k 2>&1 | tail -1
PYTHONPATH=. $M/.venv/bin/python $M/scripts/gpu_capped.py --gpu-fraction 0.75 --gpu-util 0.7 \
  $D/heldout_fit.py $CK $M/datasets/data_free_anchors_20261112_v3/eligible.pt fit_S7_10000b 500 8 4 > $D/fit_S7_10000b.log 2>&1
echo "fit exit $?"
python3 $R/analysis_scripts/scan_cache.py --evict $M/.venv/lib/python3.12/site-packages $M/datasets/data_free_anchors_20261112_v3 2>&1 | tail -1
PYTHONPATH=. $M/.venv/bin/python $M/scripts/gpu_capped.py --gpu-fraction 0.75 --gpu-util 0.7 \
  $T/rescore_fits.py $CK fit_S7_10000b $M/figures/data_free_worst_unseen_by_surface_S7_10000.png > $T/rescore_S7.out 2>&1
echo "rescore exit $?"
echo "done $(date)"
