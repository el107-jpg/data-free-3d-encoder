#!/bin/bash
# Evaluate a weight-normalisation run next to its control.
# usage: eval_wn.sh <gpu fraction> <tag> <checkpoint> <control checkpoint>
M=/home/eric-luo/Downloads/pointcloud-motion-planning
T=/home/eric-luo/.claude/jobs/8dde3c62/tmp
F=$M/runs/data_free_latent/analysis_scripts/followup
FRAC=$1; TAG=$2; CKPT=$3; CCKPT=$4
cd $M/.claude/worktrees/data-free-connectivity-plan || exit 1
run() { PYTHONPATH=. $M/.venv/bin/python $M/scripts/gpu_capped.py --gpu-fraction $FRAC --gpu-util 0.7 "$@" 2>&1 | grep -v "^\[gpu_capped\]"; }
echo "== evalrun"; run $F/evalrun.py "$CKPT" "$TAG" | tail -n 6
echo "== variance on 96 codes"; run $T/variance_noise.py "$CCKPT" "$CKPT"
echo "== gradient by layer"; run $T/grad_by_layer.py "$CCKPT" "$CKPT"
echo "== unit activity"; run $T/unit_activity.py "$CCKPT" "$CKPT"
echo "== field quality"; run $T/field_quality.py "$CKPT"
echo "== sheets"; run $T/decoded_sheet.py "$CKPT" "$TAG" | tail -n 3
echo "== done $(date)"
