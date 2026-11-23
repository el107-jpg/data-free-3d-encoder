#!/bin/bash
# S10: S9 (distilled base-free decoder at 12,000, low learning rate) continued to 20,000 at the same rate (6.364e-5),
# to see whether the limits of the sphere-residual start keep fading with more training. Evaluation follows.
M=/home/eric-luo/Downloads/pointcloud-motion-planning
T=/home/eric-luo/.claude/jobs/8dde3c62/tmp
F=$M/runs/data_free_latent/analysis_scripts/followup
R=$M/runs/data_free_latent
S10=S10_S9_continued_to_20k
python3 $R/analysis_scripts/scan_cache.py --evict $M/.venv/lib/python3.12/site-packages /home/eric-luo/.local/share/uv/python \
  $M/.claude/worktrees/data-free-connectivity-plan/arm_planner $M/.claude/worktrees/data-free-connectivity-plan/scripts \
  $R/S9_S8_lower_lr_to_12k $M/datasets/data_free_anchors_procedural_thick_plus_thin_20261113_v6 2>&1 | tail -1
setsid -f nohup bash $T/snapshot_wait.sh $S10 14000 16000 18000 20000 > $T/S10_snap.out 2>&1 < /dev/null
bash $F/cont_run.sh - $R/S9_S8_lower_lr_to_12k/checkpoint_12000.pt $S10 20000 --set lr=6.364e-5 --set diag_every=250
echo "training done $(date)"
bash $T/eval_s10.sh > $T/eval_S10.out 2>&1
echo "evaluation done $(date)"
