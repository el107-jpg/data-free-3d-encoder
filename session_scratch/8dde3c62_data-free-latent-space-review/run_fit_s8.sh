#!/bin/bash
# Unseen-object fit of S8 at 10,000 (the first try ran out of memory under the small share), then its scoring.
M=/home/eric-luo/Downloads/pointcloud-motion-planning
T=/home/eric-luo/.claude/jobs/8dde3c62/tmp
R=$M/runs/data_free_latent
D=$R/analysis_scripts/followup
CK=$R/S8_distilled_no_base_continued_to_10k/checkpoint_10000.pt
cd $M/.claude/worktrees/data-free-connectivity-plan || exit 1
python3 $R/analysis_scripts/scan_cache.py --evict $M/.venv/lib/python3.12/site-packages /home/eric-luo/.local/share/uv/python \
  $M/.claude/worktrees/data-free-connectivity-plan/arm_planner $M/datasets/data_free_anchors_20261112_v3 $R/S8_distilled_no_base_continued_to_10k 2>&1 | tail -1
mv $D/fit_S8_10000.log $D/fit_S8_10000_out_of_memory.log 2>/dev/null
PYTHONPATH=. $M/.venv/bin/python $M/scripts/gpu_capped.py --gpu-fraction 0.6 --gpu-util 0.7 \
  $D/heldout_fit.py $CK $M/datasets/data_free_anchors_20261112_v3/eligible.pt fit_S8_10000 500 8 4 > $D/fit_S8_10000.log 2>&1
echo "fit exit $?"
PYTHONPATH=. $M/.venv/bin/python $M/scripts/gpu_capped.py --gpu-fraction 0.6 --gpu-util 0.7 \
  $T/rescore_fits.py $CK fit_S8_10000 $M/figures/data_free_worst_unseen_by_surface_S8_10000.png > $T/rescore_S8.out 2>&1
echo "rescore exit $?"
echo "done $(date)"
