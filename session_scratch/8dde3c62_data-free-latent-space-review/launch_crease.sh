#!/bin/bash
# C1: J4 recipe from scratch for 1,000 updates with the crease term (weight 10).
# C2: J7 at 11,000 continued to 12,000 at J7's learning rate with the crease term (weight 10); waits for C1.
M=/home/eric-luo/Downloads/pointcloud-motion-planning
T=/home/eric-luo/.claude/jobs/8dde3c62/tmp
F=$M/runs/data_free_latent/analysis_scripts/followup
R=$M/runs/data_free_latent
C1=C1_J4_recipe_crease10_1k
C2=C2_J7_11000_crease10_to_12k
ANCHORS=$M/datasets/data_free_anchors_procedural_thick_plus_thin_20261113_v6/anchors.pt setsid -f nohup bash $F/queue_run2.sh - $C1 1000 \
  --set latent_dim=64 --set tolerances.consistency=8e-3 --set anchor_embed_lr=0.0 --set correction_bound_inner=1.35 \
  --set origin_margin=-10.0 --set consistency_fixed_surface=True --set consistency_min_depth=0.04 --set field_moves_surface=True \
  --set mult_ranges.consistency=1,20 --set anchor_exempt_size=True --set code_isotropy_weight=100.0 --set mult_ranges.pose_scale=1,500 \
  --set activation_leak=0.05 --set softplus_beta=100.0 --set fourier_bands=3 --set anchor_codes_per_pass=6 \
  --set crease_weight=10.0 > $T/C1_launch.out 2>&1 < /dev/null
setsid -f nohup bash $F/cont_run.sh $C1 $R/J7_J6_lower_lr_to_12k/checkpoint_11000.pt $C2 12000 \
  --set lr=6.364e-5 --set diag_every=250 --set crease_weight=10.0 > $T/C2_launch.out 2>&1 < /dev/null
setsid -f nohup bash $T/snapshot_at.sh $C1 500 1000 > $T/C1_snap.out 2>&1 < /dev/null
echo launched
