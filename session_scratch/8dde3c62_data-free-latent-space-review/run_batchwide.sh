#!/bin/bash
# After the pooled run finishes: batch-wide neighbour search (every shape of the update compared
# with all the others, exact gradient), learning rate 1.5x and constant, 2500 updates from scratch.
D=/home/eric-luo/.claude/jobs/8dde3c62/tmp/df_train
while pgrep -f "train_data_free_latent.py --output-dir $D/s_global_pool128" > /dev/null; do sleep 20; done
cd /home/eric-luo/Downloads/pointcloud-motion-planning/.claude/worktrees/data-free-connectivity-plan || exit 1
PY=/home/eric-luo/Downloads/pointcloud-motion-planning/.venv/bin/python
CAP="/home/eric-luo/Downloads/pointcloud-motion-planning/scripts/gpu_capped.py --gpu-fraction 0.75 --gpu-util 0.7"
O=$D/t_batchwide128
mkdir -p $O
ln -sfn $O/tb /home/eric-luo/Downloads/pointcloud-motion-planning/runs/tensorboard_active/data_free_t_batchwide128_live
PYTHONPATH=. $PY $CAP scripts/train_data_free_latent.py --output-dir $O --steps 2500 --device cuda --seed 0 \
  --set accumulate=4 --set lr=6.364e-4 --set lr_warmup_steps=125 --set lr_floor_fraction=1.0 \
  --set mult_rate=0.08 --set mult_rates.bridge=0.2 --set mult_rates.delete=0.2 --set mult_rates.fill=0.2 \
  --set mult_rates.widen=0.2 --set mult_rates.thicken=0.2 --set mult_rates.barrier=0.2 \
  --set diag_every=125 --set ckpt_every=250 --set log_every=25 --set diag_resolution=96 \
  --set clusters=4 --set codes_per_cluster=8 --set global_codes=True --set diversity_mode=nearest \
  --set spread_weight=0.0 --set batch_neighbours=True --set pool_size=0 --set pool_points=1024 --set nearest_k=8 \
  > $D/t.log 2>&1
