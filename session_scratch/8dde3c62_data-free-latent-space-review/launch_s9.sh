#!/bin/bash
# S9: S8 (distilled base-free decoder) continued from 10,000 to 12,000 at a tenth of the learning rate (6.364e-5),
# as J7 did for J6. Control: J7 (J6 continued from 10,000 to 12,000 at the same rate). Evaluation follows.
M=/home/eric-luo/Downloads/pointcloud-motion-planning
T=/home/eric-luo/.claude/jobs/8dde3c62/tmp
F=$M/runs/data_free_latent/analysis_scripts/followup
R=$M/runs/data_free_latent
S9=S9_S8_lower_lr_to_12k
python3 $R/analysis_scripts/scan_cache.py --evict $M/.venv/lib/python3.12/site-packages /home/eric-luo/.local/share/uv/python \
  $M/.claude/worktrees/data-free-connectivity-plan/arm_planner $M/.claude/worktrees/data-free-connectivity-plan/scripts \
  $R/S8_distilled_no_base_continued_to_10k $M/datasets/data_free_anchors_procedural_thick_plus_thin_20261113_v6 2>&1 | tail -1
setsid -f nohup bash $T/snapshot_wait.sh $S9 10500 11000 11500 12000 > $T/S9_snap.out 2>&1 < /dev/null
bash $F/cont_run.sh - $R/S8_distilled_no_base_continued_to_10k/checkpoint_10000.pt $S9 12000 --set lr=6.364e-5 --set diag_every=250
echo "training done $(date)"
bash $T/eval_s9.sh > $T/eval_S9.out 2>&1
echo "evaluation done $(date)"
