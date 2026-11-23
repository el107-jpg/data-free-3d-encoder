#!/bin/bash
# Wait until a run has saved a kept checkpoint (or has ended), then print its latest probe lines.
# usage: wait_for.sh <run name> [update]     (without an update: wait for the run to end)
R=/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/$1
while true; do
  if [ -n "$2" ] && [ -e "$R/checkpoint_$2.pt" ]; then break; fi
  if [ -e "$R/train.log" ] && ! pgrep -f "train_data_free_latent.py --output-dir .*/$1 " > /dev/null; then
    sleep 5
    pgrep -f "train_data_free_latent.py --output-dir .*/$1 " > /dev/null || break
  fi
  sleep 60
done
echo "$1: $([ -n "$2" ] && [ -e "$R/checkpoint_$2.pt" ] && echo "checkpoint $2 saved" || echo "run ended")"
grep -E "^diag" "$R/train.log" | tail -n 4 | cut -c1-240
tail -n 2 "$R/train.log" | cut -c1-200
