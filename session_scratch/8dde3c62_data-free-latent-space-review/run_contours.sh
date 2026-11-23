#!/bin/bash
# Field contour charts for the sphere-start runs and the J4 reference (same three random codes in each).
M=/home/eric-luo/Downloads/pointcloud-motion-planning
T=/home/eric-luo/.claude/jobs/8dde3c62/tmp
R=$M/runs/data_free_latent
F=$M/figures
S1=$R/S1_J4_recipe_sphere_start_no_base_1k
S3=$R/S3_J4_recipe_sphere_start_usual_spread_anchor_ramp_1k
J4=$R/J4_d64_nocentre_procedural96_even_code_leak05_beta100_fourier3_5k
cd $M/.claude/worktrees/data-free-connectivity-plan || exit 1
run() { PYTHONPATH=. $M/.venv/bin/python $M/scripts/gpu_capped.py --gpu-fraction 0.75 --gpu-util 0.7 $T/field_contours3.py "$@" 2>&1 | grep -v "^\[gpu_capped\]"; }
run "fresh:$S1/checkpoint_last.pt"    $F/data_free_field_sphere_start_S1_update0.png   "S1 (no fixed sphere, fitted start), before training"
run "fresh:$S1/checkpoint_last.pt@25" $F/data_free_field_sphere_start_S1_update25.png  "S1 (no fixed sphere, fitted start) after 25 updates (re-run from the start)"
run $S1/checkpoint_500.pt             $F/data_free_field_sphere_start_S1_update500.png "S1 (no fixed sphere, fitted start)"
run $S1/checkpoint_last.pt            $F/data_free_field_sphere_start_S1_update1000.png "S1 (no fixed sphere, fitted start)"
run $R/S2_J4_recipe_sphere_start_no_base_lr_third_1k/checkpoint_last.pt $F/data_free_field_sphere_start_S2_update1000.png "S2 (as S1 at one third of the learning rate)"
run $S3/checkpoint_300.pt             $F/data_free_field_sphere_start_S3_update300.png "S3 (no fixed sphere, usual code spread, anchor ramp)"
run $S3/checkpoint_last.pt            $F/data_free_field_sphere_start_S3_update1000.png "S3 (no fixed sphere, usual code spread, anchor ramp)"
run "fresh:$J4/checkpoint_1000.pt@25" $F/data_free_field_fixed_base_J4_update25.png    "J4 recipe (sphere as fixed base) after 25 updates (re-run from the start)"
run $J4/checkpoint_1000.pt            $F/data_free_field_fixed_base_J4_update1000.png  "J4 (sphere as fixed base)"
echo "done $(date)"
