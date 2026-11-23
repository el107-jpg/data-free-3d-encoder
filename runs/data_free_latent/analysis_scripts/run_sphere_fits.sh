#!/bin/bash
# Fit the plain main network to a sphere, with 3 Fourier bands (the J4 recipe) and without.
M=/home/eric-luo/Downloads/pointcloud-motion-planning
T=/home/eric-luo/.claude/jobs/8dde3c62/tmp
O=$M/runs/data_free_latent/S0_sphere_fit_20261114
CFG=$M/runs/data_free_latent/J4_d64_nocentre_procedural96_even_code_leak05_beta100_fourier3_5k/config.json
cd $M/.claude/worktrees/data-free-connectivity-plan || exit 1
for b in 3 0; do
  PYTHONPATH=. $M/.venv/bin/python $M/scripts/gpu_capped.py --gpu-fraction 0.75 --gpu-util 0.7 \
    $T/sphere_fit.py $CFG $O/fourier$b $b 4000 > $O/fourier$b.log 2>&1
done
echo "done $(date)"
