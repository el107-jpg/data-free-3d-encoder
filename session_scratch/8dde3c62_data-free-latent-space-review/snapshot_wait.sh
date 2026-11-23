#!/bin/bash
# As snapshot_at.sh, for a run that is still queued: waits for the run's log to appear first.
# usage: snapshot_wait.sh <run folder name> <update> [<update> ...]
O=/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/$1
until [ -e "$O/train.log" ]; do sleep 20; done
sleep 30
exec bash /home/eric-luo/.claude/jobs/8dde3c62/tmp/snapshot_at.sh "$@"
