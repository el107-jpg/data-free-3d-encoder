#!/bin/bash
# S5: the base-free decoder (S4: fitted sphere start, small code spread, outer limit by minimum, normal learning
# rate) with the anchor pull fully off for 500 updates and then ramped in over 500, and the slope term counting
# the near-surface samples as well. 2,500 updates so that J4 at 1,000 and at 2,500 are both controls.
M=/home/eric-luo/Downloads/pointcloud-motion-planning
T=/home/eric-luo/.claude/jobs/8dde3c62/tmp
F=$M/runs/data_free_latent/analysis_scripts/followup
S5=S5_J4_recipe_sphere_start_no_base_anchor_hold_surface_slope_2500
ANCHORS=$M/datasets/data_free_anchors_procedural_thick_plus_thin_20261113_v6/anchors.pt setsid -f nohup bash $F/queue_run2.sh - $S5 2500 \
  --set latent_dim=64 --set tolerances.consistency=8e-3 --set anchor_embed_lr=0.0 --set correction_bound_inner=1.35 \
  --set origin_margin=-10.0 --set consistency_fixed_surface=True --set consistency_min_depth=0.04 --set field_moves_surface=True \
  --set mult_ranges.consistency=1,20 --set anchor_exempt_size=True --set code_isotropy_weight=100.0 --set mult_ranges.pose_scale=1,500 \
  --set activation_leak=0.05 --set softplus_beta=100.0 --set fourier_bands=3 --set anchor_codes_per_pass=6 \
  --set sphere_base=False --set init_main_from=$M/runs/data_free_latent/S0_sphere_fit_20261114/fourier3/sphere_fit.pt \
  --set film_init_std=0.005 --set outer_limit=min \
  --set anchor_weight_start=0.0 --set anchor_weight_hold_steps=500 --set anchor_weight_ramp_steps=500 \
  --set eikonal_all_points=True > $T/S5_launch.out 2>&1 < /dev/null
setsid -f nohup bash $T/snapshot_wait.sh $S5 500 1000 2000 2500 > $T/S5_snap.out 2>&1 < /dev/null
echo launched
