#!/bin/bash
# Two runs one after the other, same settings except the diversity loss:
#   k_ordinal64 : ordinal mode (floor + order of distances), no spread term
#   l_slope64   : the original slope band, no spread term (baseline in the 4x8 cluster layout)
cd /home/eric-luo/Downloads/pointcloud-motion-planning/.claude/worktrees/data-free-connectivity-plan || exit 1
D=/home/eric-luo/.claude/jobs/8dde3c62/tmp/df_train
PY=/home/eric-luo/Downloads/pointcloud-motion-planning/.venv/bin/python
CAP="/home/eric-luo/Downloads/pointcloud-motion-planning/scripts/gpu_capped.py --gpu-fraction 0.75 --gpu-util 0.7"
COMMON="--steps 5000 --device cuda --seed 0 --set accumulate=2 --set lr=4.243e-4 --set lr_warmup_steps=250 --set mult_rate=0.04 --set mult_rates.bridge=0.1 --set mult_rates.delete=0.1 --set mult_rates.fill=0.1 --set mult_rates.widen=0.1 --set mult_rates.thicken=0.1 --set mult_rates.barrier=0.1 --set diag_every=250 --set ckpt_every=500 --set log_every=50 --set clusters=4 --set codes_per_cluster=8 --set spread_weight=0.0"
mkdir -p $D/k_ordinal64 $D/l_slope64
PYTHONPATH=. $PY $CAP scripts/train_data_free_latent.py --output-dir $D/k_ordinal64 $COMMON --set diversity_mode=ordinal > $D/k.log 2>&1
PYTHONPATH=. $PY $CAP scripts/train_data_free_latent.py --output-dir $D/l_slope64 $COMMON --set diversity_mode=slope > $D/l.log 2>&1
