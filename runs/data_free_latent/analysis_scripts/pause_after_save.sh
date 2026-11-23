#!/bin/bash
# Stop a run right after its next checkpoint save.
# usage: pause_after_save.sh <run name>
O=/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/$1
PAT="train_data_free_latent.py --output-dir .*/$1 "
before=$(stat -c %Y "$O/checkpoint_last.pt")
until [ "$(stat -c %Y "$O/checkpoint_last.pt")" != "$before" ]; do sleep 2; done
sleep 3   # let the write (temporary file then rename) settle
pkill -f "snapshot_both.sh $1 "
pkill -f "cont_run.sh .* $1 "
pkill -f "$PAT"
sleep 5
pgrep -af "$PAT" > /dev/null && echo "still running" || echo "stopped"
python3 -c "
import json
rows = [json.loads(l) for l in open('$O/metrics.jsonl') if l.strip()]
print('last logged update', rows[-1]['step'])
"
date
