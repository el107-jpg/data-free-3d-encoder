#!/bin/bash
# What is training, what is queued, and how far the anchor build is.
R=/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent
date
ps -eo pid,etimes,args | grep -E "scripts/train_data_free_latent.py|build_data_free_anchors.py|queue_run2.sh|cont_run.sh|snapshot_at.sh" | grep -v grep | grep -v "bash -c" \
  | sed -E 's/--set .*//' | cut -c1-170
nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader
for n in "$@"; do
  echo "== $n: $(ls $R/$n 2>/dev/null | grep -c checkpoint_) checkpoints"
  grep -E "^diag" $R/$n/train.log 2>/dev/null | tail -n 1 | cut -c1-240 | sed -E 's/\(1 piece.*graph=grid [0-9.]+//'
  tail -n 1 $R/$n/train.log 2>/dev/null | cut -c1-100
done
tail -n 2 /home/eric-luo/.claude/jobs/8dde3c62/tmp/anchor_build_v3.log | cut -c1-300
