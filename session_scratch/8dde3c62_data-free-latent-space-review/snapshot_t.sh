#!/bin/bash
# Keep a copy of the long run's checkpoint every 500 updates (the trainer only keeps the latest).
O=/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/t_batchwide128
PAT="train_data_free_latent.py --output-dir .*t_batchwide128 --steps 20000"
last=""
while pgrep -f "$PAT" > /dev/null; do
  m=$(stat -c %Y "$O/checkpoint_last.pt" 2>/dev/null)
  if [ -n "$m" ] && [ "$m" != "$last" ]; then
    last=$m
    s=$(tail -n 1 "$O/metrics.jsonl" | sed -E 's/.*"step": ([0-9]+).*/\1/')
    if [[ "$s" =~ ^[0-9]+$ ]]; then
      r=$(( (s + 50) / 100 * 100 ))
      if [ $(( r % 500 )) -eq 0 ] && [ ! -e "$O/checkpoint_$r.pt" ]; then
        cp "$O/checkpoint_last.pt" "$O/checkpoint_$r.pt"
      fi
    fi
  fi
  sleep 20
done
echo "trainer no longer running; snapshots:"; ls "$O" | grep -c "checkpoint_[0-9]"
