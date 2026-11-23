#!/bin/bash
# Continue a saved model in a NEW run folder with the code in THIS worktree (the source checkpoint is
# copied, never modified).
# usage: cont_run.sh <run to wait for, or -> <source checkpoint> <new run name> <total updates> [--set key=value ...]
# The source checkpoint is read after the wait, so it can be the final checkpoint of the run waited for.
# Same arguments as runs/data_free_latent/analysis_scripts/followup/cont_run.sh, so monitor/status.py lists it.
# The trainer is started through run_guarded.sh (memory-fault scan, and a retry if it dies before training).
M=/home/eric-luo/Downloads/pointcloud-motion-planning
W=$(cd "$(dirname "$0")/../.." && pwd)
WAIT=$1; SRC=$2; NAME=$3; STEPS=$4; shift 4
if [ "$WAIT" != "-" ]; then
  until [ -e "$M/runs/data_free_latent/$WAIT/train.log" ] && ! pgrep -f "train_data_free_latent.py --output-dir .*/$WAIT " > /dev/null; do
    sleep 20
  done
  sleep 30      # a failed start of the run waited for is retried within this time
  until ! pgrep -f "train_data_free_latent.py --output-dir .*/$WAIT " > /dev/null; do
    sleep 20
  done
fi
O=$M/runs/data_free_latent/$NAME
if [ -e "$O/checkpoint_last.pt" ]; then echo "$NAME already exists; refusing to overwrite"; exit 1; fi
if [ ! -e "$SRC" ]; then echo "source checkpoint $SRC is missing; $NAME not started"; exit 1; fi
mkdir -p "$O"
cp "$SRC" "$O/checkpoint_last.pt"
cp "$SRC" "$O/checkpoint_start.pt"
ln -sfn "$O/tb" "$M/runs/tensorboard_active/data_free_$NAME"
cd "$W" || exit 1
git log --oneline -n 1 > "$O/code_version.txt"
bash "$W/scripts/data_free_queue/run_guarded.sh" "$O" "$W" "train_data_free_latent.py --output-dir $O " -- --steps "$STEPS" --device cuda --resume --set ckpt_every=100 "$@"
echo "$NAME finished: $(grep -E '^diag' "$O/train.log" | tail -n 1 | cut -c1-260)"
