#!/bin/bash
# S7: S6 (no-base decoder) continued from 5,000 to 10,000 with nothing changed (new folder; S6's checkpoint is
# copied). Snapshots every 1,000. Controls: J6 (J4 continued over the same stretch) at 6,000 / 7,500 / 9,000 / 10,000.
M=/home/eric-luo/Downloads/pointcloud-motion-planning
T=/home/eric-luo/.claude/jobs/8dde3c62/tmp
F=$M/runs/data_free_latent/analysis_scripts/followup
R=$M/runs/data_free_latent
S7=S7_S6_continued_to_10k
setsid -f nohup bash $F/cont_run.sh - $R/S6_S5_continued_to_5k/checkpoint_last.pt $S7 10000 \
  --set diag_every=250 > $T/S7_launch.out 2>&1 < /dev/null
setsid -f nohup bash $T/snapshot_wait.sh $S7 6000 7000 8000 9000 10000 > $T/S7_snap.out 2>&1 < /dev/null
echo launched
