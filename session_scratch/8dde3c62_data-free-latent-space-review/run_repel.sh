#!/bin/bash
# After the statistic-repulsion run finishes: from scratch, pure push (no floor, no ordering) + spread.
# Same settings as m_ordinal_spread64 except the diversity mode, for a one-variable comparison.
D=/home/eric-luo/.claude/jobs/8dde3c62/tmp/df_train
while pgrep -f "train_data_free_latent.py --output-dir $D/o_repulsion" > /dev/null; do sleep 20; done
cd /home/eric-luo/Downloads/pointcloud-motion-planning/.claude/worktrees/data-free-connectivity-plan || exit 1
PY=/home/eric-luo/Downloads/pointcloud-motion-planning/.venv/bin/python
CAP="/home/eric-luo/Downloads/pointcloud-motion-planning/scripts/gpu_capped.py --gpu-fraction 0.75 --gpu-util 0.7"
COMMON="--steps 5000 --device cuda --seed 0 --set accumulate=2 --set lr=4.243e-4 --set lr_warmup_steps=250 --set mult_rate=0.04 --set mult_rates.bridge=0.1 --set mult_rates.delete=0.1 --set mult_rates.fill=0.1 --set mult_rates.widen=0.1 --set mult_rates.thicken=0.1 --set mult_rates.barrier=0.1 --set diag_every=250 --set ckpt_every=500 --set log_every=50 --set clusters=4 --set codes_per_cluster=8 --set mult_ranges.pose_axes=1,50"
mkdir -p $D/p_repel_spread64
ln -sfn $D/p_repel_spread64/tb /home/eric-luo/Downloads/pointcloud-motion-planning/runs/tensorboard_active/data_free_p_repel_spread64_live
PYTHONPATH=. $PY $CAP scripts/train_data_free_latent.py --output-dir $D/p_repel_spread64 $COMMON --set diversity_mode=repel --set spread_weight=1.0 > $D/p.log 2>&1
