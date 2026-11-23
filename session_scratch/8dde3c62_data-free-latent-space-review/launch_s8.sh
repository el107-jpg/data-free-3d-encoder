#!/bin/bash
# D0: distil J4 at 5,000 into the base-free decoder (J4's weights as the start), then
# S8: continue that model from 5,000 to 10,000 with the J4 recipe's losses (no base, outer limit by minimum,
#     slope term counting the near-surface samples). Control: J6 (J4 continued over the same stretch, with the base).
M=/home/eric-luo/Downloads/pointcloud-motion-planning
T=/home/eric-luo/.claude/jobs/8dde3c62/tmp
F=$M/runs/data_free_latent/analysis_scripts/followup
R=$M/runs/data_free_latent
D=$R/D0_distill_J4_5000_no_base
mkdir -p $D
cd $M/.claude/worktrees/data-free-connectivity-plan || exit 1
python3 $R/analysis_scripts/scan_cache.py --evict $M/.venv/lib/python3.12/site-packages /home/eric-luo/.local/share/uv/python \
  $M/.claude/worktrees/data-free-connectivity-plan/arm_planner $M/.claude/worktrees/data-free-connectivity-plan/scripts \
  $R/J4_d64_nocentre_procedural96_even_code_leak05_beta100_fourier3_5k $M/datasets/data_free_anchors_procedural_thick_plus_thin_20261113_v6 2>&1 | tail -1
PYTHONPATH=. $M/.venv/bin/python $M/scripts/gpu_capped.py --gpu-fraction 0.75 --gpu-util 0.7 \
  scripts/distill_data_free.py --teacher $R/J4_d64_nocentre_procedural96_even_code_leak05_beta100_fourier3_5k/checkpoint_last.pt \
  --output $D/distilled_5000.pt --steps 5000 --lr 3e-4 --device cuda \
  --set sphere_base=False --set outer_limit=min --set eikonal_all_points=True > $D/distill.log 2>&1
echo "distill exit $? $(date)"
[ -e $D/distilled_5000.pt ] || exit 1
setsid -f nohup bash $T/snapshot_wait.sh S8_distilled_no_base_continued_to_10k 6000 7000 8000 9000 10000 > $T/S8_snap.out 2>&1 < /dev/null
bash $F/cont_run.sh - $D/distilled_5000.pt S8_distilled_no_base_continued_to_10k 10000 --set diag_every=250
echo "continuation done $(date)"
