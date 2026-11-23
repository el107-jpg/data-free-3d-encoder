#!/bin/bash
# Full evaluation of a sphere-start run: standard measures, facing, field error over the run, gradient
# breakdown, contour charts.   usage: eval_s4.sh <tag> <run folder name>
M=/home/eric-luo/Downloads/pointcloud-motion-planning
T=/home/eric-luo/.claude/jobs/8dde3c62/tmp
R=$M/runs/data_free_latent
TAG=$1; RUN=$R/$2
cd $M/.claude/worktrees/data-free-connectivity-plan || exit 1
run() { PYTHONPATH=. $M/.venv/bin/python $M/scripts/gpu_capped.py --gpu-fraction 0.75 --gpu-util 0.7 "$@" 2>&1 | grep -v "^\[gpu_capped\]"; }
cat $T/${TAG}_launch.out
grep -E "^diag" $RUN/train.log | cut -c1-215
bash $T/eval_sphere_run.sh 0.75 ${TAG}_1000 $RUN/checkpoint_last.pt
echo "== facing"; run $T/facing.py $RUN/checkpoint_last.pt
echo "== field error over the run"; python3 $T/field_error_over_run.py $TAG=$RUN
echo "off $(grep -c '"graph_mode": "off"' $RUN/metrics.jsonl) sign $(grep -c '"graph_mode": "sign"' $RUN/metrics.jsonl) full $(grep -c '"graph_mode": "full"' $RUN/metrics.jsonl)"
echo "== variance at 500"; run $T/variance_noise.py $RUN/checkpoint_500.pt
echo "== gradient by term"; run $T/grad_by_term.py $RUN/checkpoint_500.pt $RUN/checkpoint_last.pt
echo "== contours"
run $T/field_contours3.py "fresh:$RUN/checkpoint_last.pt@25" $M/figures/data_free_field_sphere_start_${TAG}_update25.png "$TAG (no fixed sphere, fitted start, outer limit by minimum) after 25 updates (re-run from the start)"
run $T/field_contours3.py $RUN/checkpoint_last.pt $M/figures/data_free_field_sphere_start_${TAG}_update1000.png "$TAG (no fixed sphere, fitted start, outer limit by minimum)"
echo "== done $(date)"
