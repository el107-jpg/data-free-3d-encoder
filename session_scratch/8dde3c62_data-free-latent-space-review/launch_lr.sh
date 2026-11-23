#!/bin/bash
# L1: J4 continued from 5,000 to 10,000 with the final layer at a tenth of the learning rate (nothing else
#     changed); a running weight average is kept on the side and does not affect training. Control: J6.
# L2: the same plus learning-rate decay starting at 5,000 (cosine down to a tenth by 10,000); waits for L1.
M=/home/eric-luo/Downloads/pointcloud-motion-planning
T=/home/eric-luo/.claude/jobs/8dde3c62/tmp
F=$M/runs/data_free_latent/analysis_scripts/followup
R=$M/runs/data_free_latent
SRC=$R/J4_d64_nocentre_procedural96_even_code_leak05_beta100_fourier3_5k/checkpoint_last.pt
L1=L1_J4_5000_final_layer_lr_tenth_to_10k
L2=L2_J4_5000_final_layer_lr_tenth_decay_to_10k
setsid -f nohup bash $F/cont_run.sh - $SRC $L1 10000 --set diag_every=250 \
  --set out_lr_scale=0.1 --set weight_average_decay=0.998 > $T/L1_launch.out 2>&1 < /dev/null
setsid -f nohup bash $T/snapshot_both.sh $L1 6000 7500 9000 10000 > $T/L1_snap.out 2>&1 < /dev/null
setsid -f nohup bash $F/cont_run.sh $L1 $SRC $L2 10000 --set diag_every=250 \
  --set out_lr_scale=0.1 --set weight_average_decay=0.998 \
  --set lr_decay_start=5000 --set lr_floor_fraction=0.1 > $T/L2_launch.out 2>&1 < /dev/null
setsid -f nohup bash $T/snapshot_both.sh $L2 6000 7500 9000 10000 > $T/L2_snap.out 2>&1 < /dev/null
echo launched
