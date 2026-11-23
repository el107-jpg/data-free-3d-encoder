#!/usr/bin/env bash
set -uo pipefail
setsid -f nohup /home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/queue_logs/launch_B4_scratch_box_linear_sepnet_budget02_1k.sh > /home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/queue_logs/launch_B4_scratch_box_linear_sepnet_budget02_1k.out 2>&1 < /dev/null
until [ -f /home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/queue_logs/B4_scratch_box_linear_sepnet_budget02_1k.pid ] && pgrep -f "output-dir /home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/B4_scratch" > /dev/null; do sleep 10; done
exec /home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/queue_logs/launch_B6_scratch_box_linear_sharedmlp_budget02_1k.sh > /home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/queue_logs/launch_B6_scratch_box_linear_sharedmlp_budget02_1k.out 2>&1
