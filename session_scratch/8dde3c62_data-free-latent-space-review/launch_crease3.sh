#!/bin/bash
# C3: as C1 (J4 recipe from scratch, 1,000 updates) with the crease term at weight 2 instead of 10; waits for C2.
M=/home/eric-luo/Downloads/pointcloud-motion-planning
T=/home/eric-luo/.claude/jobs/8dde3c62/tmp
F=$M/runs/data_free_latent/analysis_scripts/followup
C3=C3_J4_recipe_crease2_1k
ANCHORS=$M/datasets/data_free_anchors_procedural_thick_plus_thin_20261113_v6/anchors.pt setsid -f nohup bash $F/queue_run2.sh C2_J7_11000_crease10_to_12k $C3 1000 \
  --set latent_dim=64 --set tolerances.consistency=8e-3 --set anchor_embed_lr=0.0 --set correction_bound_inner=1.35 \
  --set origin_margin=-10.0 --set consistency_fixed_surface=True --set consistency_min_depth=0.04 --set field_moves_surface=True \
  --set mult_ranges.consistency=1,20 --set anchor_exempt_size=True --set code_isotropy_weight=100.0 --set mult_ranges.pose_scale=1,500 \
  --set activation_leak=0.05 --set softplus_beta=100.0 --set fourier_bands=3 --set anchor_codes_per_pass=6 \
  --set crease_weight=2.0 > $T/C3_launch.out 2>&1 < /dev/null
echo launched
