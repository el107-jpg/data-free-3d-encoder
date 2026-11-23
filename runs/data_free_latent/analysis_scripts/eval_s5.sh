#!/bin/bash
# Evaluate S5 at 1,000 and 2,500 beside another training (small GPU share).
M=/home/eric-luo/Downloads/pointcloud-motion-planning
T=/home/eric-luo/.claude/jobs/8dde3c62/tmp
F=$M/runs/data_free_latent/analysis_scripts/followup
RUN=$M/runs/data_free_latent/S5_J4_recipe_sphere_start_no_base_anchor_hold_surface_slope_2500
cd $M/.claude/worktrees/data-free-connectivity-plan || exit 1
run() { PYTHONPATH=. $M/.venv/bin/python $M/scripts/gpu_capped.py --gpu-fraction 0.2 --gpu-util 0.4 "$@" 2>&1 | grep -v "^\[gpu_capped\]"; }
cat $T/S5_launch.out
grep -E "^diag" $RUN/train.log | cut -c1-215
ls $RUN | grep checkpoint
for u in 1000 2500; do
  c=$RUN/checkpoint_$u.pt; [ "$u" = 2500 ] && c=$RUN/checkpoint_last.pt
  echo "== $u evalrun"; run $F/evalrun.py $c S5_$u | tail -n 5
  echo "== $u variance"; run $T/variance_noise.py $c
  echo "== $u field quality"; run $T/field_quality.py $c
  echo "== $u facing"; run $T/facing.py $c
done
echo "== variance at 500 and 2000"; run $T/variance_noise.py $RUN/checkpoint_500.pt $RUN/checkpoint_2000.pt
echo "== field error over the run"
python3 - <<'EOF'
import json, statistics as st
rows = [json.loads(l) for l in open("/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/S5_J4_recipe_sphere_start_no_base_anchor_hold_surface_slope_2500/metrics.jsonl") if l.strip()]
for a, b in ((0, 250), (250, 500), (500, 750), (750, 1000), (1000, 1250), (1250, 1500), (1500, 1750), (1750, 2000), (2000, 2250), (2250, 2500)):
    seg = [r for r in rows if a <= r["step"] < b]
    e = sorted(r["e"] for r in seg)
    off = sum(r["graph_mode"] == "off" for r in seg) / len(seg)
    print(f"  {a:4d}-{b:4d}: field error median {st.median(e):.3f} (90%: {e[9 * len(e) // 10]:.3f}); topology terms off {off:.0%}; variety loss {st.mean(r['diversity'] for r in seg):.1f}; "
          f"slope weight {st.mean(r['mult/eikonal'] for r in seg):.0f}; anchor weight {st.mean(r.get('anchor/weight', 0) for r in seg):.0f}; anchor overlap in batch {st.mean(r.get('anchor/iou', 0) for r in seg):.2f}; "
          f"wall/gap sites {st.mean(r.get('graph/wall_sites', 0) for r in seg):.0f}/{st.mean(r.get('graph/gap_sites', 0) for r in seg):.0f}; thicken weight {st.mean(r['mult/thicken'] for r in seg):.0f}")
EOF
echo "== sheets and contours"
run $T/decoded_sheet.py $RUN/checkpoint_last.pt S5_2500 | tail -n 2
run $T/field_contours3.py $RUN/checkpoint_1000.pt $M/figures/data_free_field_sphere_start_S5_update1000.png "S5 (no fixed sphere, anchors held off 500 updates, slope term at the surface)"
run $T/field_contours3.py $RUN/checkpoint_last.pt $M/figures/data_free_field_sphere_start_S5_update2500.png "S5 (no fixed sphere, anchors held off 500 updates, slope term at the surface)"
echo "== done $(date)"
