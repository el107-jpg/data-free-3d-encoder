#!/bin/bash
# Keep copies of a run's checkpoint at chosen update counts (the trainer only keeps the latest).
# usage: snapshot_at.sh <run folder name> <update> [<update> ...]
O=/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/$1
shift
PAT="train_data_free_latent.py --output-dir .*$(basename "$O") "
last=""
while true; do
  m=$(stat -c %Y "$O/checkpoint_last.pt" 2>/dev/null)
  if [ -n "$m" ] && [ "$m" != "$last" ]; then
    last=$m
    s=$(tail -n 1 "$O/metrics.jsonl" | sed -E 's/.*"step": ([0-9]+).*/\1/')
    if [[ "$s" =~ ^[0-9]+$ ]]; then
      r=$(( (s + 50) / 100 * 100 ))
      for want in "$@"; do
        if [ "$r" -eq "$want" ] && [ ! -e "$O/checkpoint_$r.pt" ]; then cp "$O/checkpoint_last.pt" "$O/checkpoint_$r.pt"; fi
      done
    fi
  fi
  done_all=1
  for want in "$@"; do [ -e "$O/checkpoint_$want.pt" ] || done_all=0; done
  [ "$done_all" -eq 1 ] && break
  if [ -e "$O/train.log" ] && ! pgrep -f "$PAT" > /dev/null; then sleep 3; pgrep -f "$PAT" > /dev/null || break; fi
  sleep 15
done
ls "$O" | grep "checkpoint_"
