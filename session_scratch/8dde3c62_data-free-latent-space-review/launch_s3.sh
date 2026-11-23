#!/bin/bash
# S3: S1 (J4 recipe without the fixed sphere, main network started from the fitted sphere, normal learning
# rate) with two changes: the code's usual starting spread (film_init_std 0.1, not 0.005), and the anchor
# pull ramped in linearly from 1% to full weight over the first 300 updates. Controls: S1 and J4 at 1,000.
M=/home/eric-luo/Downloads/pointcloud-motion-planning
T=/home/eric-luo/.claude/jobs/8dde3c62/tmp
F=$M/runs/data_free_latent/analysis_scripts/followup
S3=S3_J4_recipe_sphere_start_usual_spread_anchor_ramp_1k
ANCHORS=$M/datasets/data_free_anchors_procedural_thick_plus_thin_20261113_v6/anchors.pt setsid -f nohup bash $F/queue_run2.sh - $S3 1000 \
  --set latent_dim=64 --set tolerances.consistency=8e-3 --set anchor_embed_lr=0.0 --set correction_bound_inner=1.35 \
  --set origin_margin=-10.0 --set consistency_fixed_surface=True --set consistency_min_depth=0.04 --set field_moves_surface=True \
  --set mult_ranges.consistency=1,20 --set anchor_exempt_size=True --set code_isotropy_weight=100.0 --set mult_ranges.pose_scale=1,500 \
  --set activation_leak=0.05 --set softplus_beta=100.0 --set fourier_bands=3 --set anchor_codes_per_pass=6 \
  --set sphere_base=False --set init_main_from=$M/runs/data_free_latent/S0_sphere_fit_20261114/fourier3/sphere_fit.pt \
  --set anchor_weight_ramp_steps=300 --set anchor_weight_start=0.01 > $T/S3_launch.out 2>&1 < /dev/null
setsid -f nohup bash $T/snapshot_wait.sh $S3 300 500 1000 > $T/S3_snap.out 2>&1 < /dev/null
echo launched
