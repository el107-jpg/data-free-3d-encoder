#!/bin/bash
# Evaluate a continuation run (live and weight-averaged checkpoints) at 6,000 / 7,500 / 9,000 / 10,000, with J6 as control.
# usage: eval_lr_run.sh <gpu fraction> <run folder> <short tag>
M=/home/eric-luo/Downloads/pointcloud-motion-planning
T=/home/eric-luo/.claude/jobs/8dde3c62/tmp
F=$M/runs/data_free_latent/analysis_scripts/followup
R=$M/runs/data_free_latent
FRAC=$1; RUN=$R/$2; TAG=$3
J6=$R/J6_J4_continued_to_10k
cd $M/.claude/worktrees/data-free-connectivity-plan || exit 1
run() { PYTHONPATH=. $M/.venv/bin/python $M/scripts/gpu_capped.py --gpu-fraction $FRAC --gpu-util 0.6 "$@" 2>&1 | grep -v "^\[gpu_capped\]"; }
for u in 6000 7500 9000 10000; do
  live=$RUN/checkpoint_$u.pt; avg=$RUN/checkpoint_average_$u.pt
  [ "$u" = 10000 ] && live=$RUN/checkpoint_last.pt && avg=$RUN/checkpoint_average_last.pt
  echo "== $u live"; run $F/evalrun.py "$live" "${TAG}_$u" | grep -E "topology|anchors"
  echo "== $u average"; run $F/evalrun.py "$avg" "${TAG}avg_$u" | grep -E "topology|anchors"
  echo "== $u variance on 96 codes (live, average)"; run $T/variance_noise.py "$live" "$avg"
done
echo "== field quality at 10,000 (live, average)"; run $T/field_quality.py $RUN/checkpoint_last.pt $RUN/checkpoint_average_last.pt
echo "== step by layer at 7,500 and 10,000"; run $T/step_by_layer.py $RUN/checkpoint_7500.pt $RUN/checkpoint_last.pt
echo "== sheets"; run $T/decoded_sheet.py $RUN/checkpoint_last.pt "${TAG}_10000" | tail -n 2; run $T/decoded_sheet.py $RUN/checkpoint_average_last.pt "${TAG}avg_10000" | tail -n 2
echo "== done $(date)"
