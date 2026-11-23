#!/bin/bash
# Unseen-object fit of the two controls for S8: J6 at 10,000 (J4 continued with the base) and the distilled copy at 5,000.
M=/home/eric-luo/Downloads/pointcloud-motion-planning
T=/home/eric-luo/.claude/jobs/8dde3c62/tmp
R=$M/runs/data_free_latent
D=$R/analysis_scripts/followup
cd $M/.claude/worktrees/data-free-connectivity-plan || exit 1
for pair in "fit_J6_10000=$R/J6_J4_continued_to_10k/checkpoint_last.pt" "fit_D0_5000=$R/D0_distill_J4_5000_no_base/distilled_5000.pt"; do
  tag=${pair%%=*}; ck=${pair#*=}
  python3 $R/analysis_scripts/scan_cache.py --evict $M/.venv/lib/python3.12/site-packages /home/eric-luo/.local/share/uv/python \
    $M/.claude/worktrees/data-free-connectivity-plan/arm_planner $M/datasets/data_free_anchors_20261112_v3 $ck 2>&1 | tail -1
  PYTHONPATH=. $M/.venv/bin/python $M/scripts/gpu_capped.py --gpu-fraction 0.6 --gpu-util 0.7 \
    $D/heldout_fit.py $ck $M/datasets/data_free_anchors_20261112_v3/eligible.pt $tag 500 8 4 > $D/$tag.log 2>&1
  echo "$tag fit exit $?"
  PYTHONPATH=. $M/.venv/bin/python $M/scripts/gpu_capped.py --gpu-fraction 0.6 --gpu-util 0.7 \
    $T/rescore_fits.py $ck $tag > $T/rescore_${tag#fit_}.out 2>&1
  echo "$tag rescore exit $?"
done
echo "done $(date)"
