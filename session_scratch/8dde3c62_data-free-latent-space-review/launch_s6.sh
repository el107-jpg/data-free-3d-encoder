#!/bin/bash
# S6: S5 continued from 2,500 to 5,000 with nothing changed (new folder; S5's checkpoint is copied). Waits for
# the other session's B1 run to finish. Control: J4 at 5,000.
M=/home/eric-luo/Downloads/pointcloud-motion-planning
T=/home/eric-luo/.claude/jobs/8dde3c62/tmp
F=$M/runs/data_free_latent/analysis_scripts/followup
R=$M/runs/data_free_latent
S6=S6_S5_continued_to_5k
setsid -f nohup bash $F/cont_run.sh B1_J4_5000_box_base_ramp200_1k \
  $R/S5_J4_recipe_sphere_start_no_base_anchor_hold_surface_slope_2500/checkpoint_last.pt $S6 5000 \
  --set diag_every=250 > $T/S6_launch.out 2>&1 < /dev/null
setsid -f nohup bash $T/snapshot_wait.sh $S6 3000 4000 5000 > $T/S6_snap.out 2>&1 < /dev/null
echo launched
