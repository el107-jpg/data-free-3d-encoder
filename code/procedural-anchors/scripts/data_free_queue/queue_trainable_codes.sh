#!/bin/bash
# Queue the trainable-anchor-code experiment behind a running run. Everything is detached, so it survives
# the session that started it. usage: queue_trainable_codes.sh <run to wait for>
#
#   TA1  the J4 recipe from scratch, 5,000 updates, with trainable anchor codes
#        (fixed for 500 updates, then their learning rate ramps to 3e-4 over 1,000)
#   TA1s TA1 continued to 6,000 at one-tenth of both learning rates (settle stage)
#   TA2  TA1 again with the surface-band variety measure (variety_surface_band=0.1)
#   TA2s TA2 continued to 6,000 the same way
M=/home/eric-luo/Downloads/pointcloud-motion-planning
R=$M/runs/data_free_latent
Q=$(cd "$(dirname "$0")" && pwd)
LOGS=$R/queue_logs
mkdir -p "$LOGS"
J4=J4_d64_nocentre_procedural96_even_code_leak05_beta100_fourier3_5k
A=TA1_J4_trainable_codes_5k
AS=TA1s_settle_6k
B=TA2_J4_trainable_codes_surface_band_5k
BS=TA2s_settle_6k
SETTLE="--set lr=6.364e-5 --set anchor_embed_lr=3e-5 --set diag_every=250"
start() { setsid nohup "$@" < /dev/null > "$LOGS/$NAME.log" 2>&1 & }

NAME=$A;  REFERENCE=$J4 start env REFERENCE=$J4 bash "$Q/queue_run2.sh" "$1" $A 5000 --set anchor_embed_lr=3e-4
NAME=$AS; start bash "$Q/cont_run.sh" $A "$R/$A/checkpoint_last.pt" $AS 6000 $SETTLE
NAME=$B;  start env REFERENCE=$J4 bash "$Q/queue_run2.sh" $AS $B 5000 --set anchor_embed_lr=3e-4 --set variety_surface_band=0.1
NAME=$BS; start bash "$Q/cont_run.sh" $B "$R/$B/checkpoint_last.pt" $BS 6000 $SETTLE
# keep the checkpoints J4 kept, plus the settle stage's halfway point
for pair in "$A 1000 2500 4000" "$AS 5500" "$B 1000 2500 4000" "$BS 5500"; do
  set -- $pair
  NAME=snapshots_$1; start bash "$R/analysis_scripts/snapshot_at.sh" "$@"
done
sleep 1
pgrep -af "queue_run2.sh|cont_run.sh|snapshot_at.sh" | grep -E "TA[12]" | cut -c1-150
