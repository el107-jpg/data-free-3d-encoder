#!/bin/bash
# Evaluate a sphere-start run next to J4 at 1,000.
# usage: eval_sphere_run.sh <gpu fraction> <tag> <checkpoint>
M=/home/eric-luo/Downloads/pointcloud-motion-planning
T=/home/eric-luo/.claude/jobs/8dde3c62/tmp
F=$M/runs/data_free_latent/analysis_scripts/followup
FRAC=$1; TAG=$2; CKPT=$3
cd $M/.claude/worktrees/data-free-connectivity-plan || exit 1
run() { PYTHONPATH=. $M/.venv/bin/python $M/scripts/gpu_capped.py --gpu-fraction $FRAC --gpu-util 0.5 "$@" 2>&1 | grep -v "^\[gpu_capped\]"; }
echo "== evalrun"; run $F/evalrun.py "$CKPT" "$TAG" | tail -n 6
echo "== variance on 96 codes"; run $T/variance_noise.py "$CKPT"
echo "== field quality"; run $T/field_quality.py "$CKPT"
echo "== sharpness"; run $T/sharpness.py $T/sharpness_$TAG.json "$TAG=$CKPT"
echo "== sheets"; run $T/decoded_sheet.py "$CKPT" "$TAG" | tail -n 2
echo "== done $(date)"
