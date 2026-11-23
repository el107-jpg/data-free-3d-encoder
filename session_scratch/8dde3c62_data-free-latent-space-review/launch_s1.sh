#!/bin/bash
# S1: the J4 recipe for 1,000 updates WITHOUT the fixed sphere in the formula; the main network starts from
# the fitted sphere (S0, 3 Fourier bands) and the code's starting spread is 0.05 x the usual so codes start
# as close together as they do today. Same learning rate. Control: J4 at 1,000.
M=/home/eric-luo/Downloads/pointcloud-motion-planning
T=/home/eric-luo/.claude/jobs/8dde3c62/tmp
F=$M/runs/data_free_latent/analysis_scripts/followup
S1=S1_J4_recipe_sphere_start_no_base_1k
ANCHORS=$M/datasets/data_free_anchors_procedural_thick_plus_thin_20261113_v6/anchors.pt setsid -f nohup bash $F/queue_run2.sh - $S1 1000 \
  --set latent_dim=64 --set tolerances.consistency=8e-3 --set anchor_embed_lr=0.0 --set correction_bound_inner=1.35 \
  --set origin_margin=-10.0 --set consistency_fixed_surface=True --set consistency_min_depth=0.04 --set field_moves_surface=True \
  --set mult_ranges.consistency=1,20 --set anchor_exempt_size=True --set code_isotropy_weight=100.0 --set mult_ranges.pose_scale=1,500 \
  --set activation_leak=0.05 --set softplus_beta=100.0 --set fourier_bands=3 --set anchor_codes_per_pass=6 \
  --set sphere_base=False --set init_main_from=$M/runs/data_free_latent/S0_sphere_fit_20261114/fourier3/sphere_fit.pt \
  --set film_init_std=0.005 > $T/S1_launch.out 2>&1 < /dev/null
setsid -f nohup bash $T/snapshot_at.sh $S1 250 500 1000 > $T/S1_snap.out 2>&1 < /dev/null
echo launched
