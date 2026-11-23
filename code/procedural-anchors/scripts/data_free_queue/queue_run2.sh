#!/bin/bash
# Start a from-scratch anchor run with the code in THIS worktree, once the GPU is free or another run has
# finished.
# usage: [SEED=n] [ANCHORS=path] [REFERENCE=run name] queue_run2.sh <run to wait for, or gpu-free, or -> <new run name> <updates> [--set key=value ...]
#   gpu-free: wait until no GPU job launched through scripts/gpu_capped.py is running (checked twice, 30 s apart).
#   REFERENCE: take every setting from that earlier run's saved config (recipe_from_run.py) instead of the
#   base list below; later --set arguments override either.
# Without REFERENCE the base settings are those of runs/data_free_latent/analysis_scripts/followup/queue_run2.sh
# (the "L2" anchor recipe). Named like that script so that monitor/status.py lists it as queued.
# The trainer is started through run_guarded.sh (memory-fault scan, and a retry if it dies before training).
M=/home/eric-luo/Downloads/pointcloud-motion-planning
W=$(cd "$(dirname "$0")/../.." && pwd)
SEED=${SEED:-0}
ANCHORS=${ANCHORS:-$M/datasets/data_free_anchors_procedural_thick_20261112_v5/anchors.pt}
WAIT=$1; NAME=$2; STEPS=$3; shift 3
busy() { pgrep -f "scripts/gpu_capped.py" > /dev/null; }
if [ "$WAIT" = "gpu-free" ]; then
  while true; do
    if busy; then sleep 30; continue; fi
    sleep 30
    busy || break
  done
elif [ "$WAIT" != "-" ]; then
  until [ -e "$M/runs/data_free_latent/$WAIT/train.log" ] && ! pgrep -f "train_data_free_latent.py --output-dir .*/$WAIT " > /dev/null; do
    sleep 20
  done
  sleep 5
fi
O=$M/runs/data_free_latent/$NAME
if [ -e "$O/train.log" ]; then echo "$NAME already exists; refusing to overwrite"; exit 1; fi
mkdir -p "$O"
ln -sfn "$O/tb" "$M/runs/tensorboard_active/data_free_$NAME"
cd "$W" || exit 1
git log --oneline -n 1 > "$O/code_version.txt"
if [ -n "$REFERENCE" ]; then
  python3 $M/runs/data_free_latent/analysis_scripts/scan_cache.py --evict $M/.venv \
    $HOME/.local/share/uv/python/cpython-3.12.16-linux-x86_64-gnu "$W/arm_planner" > /dev/null 2>&1
  mapfile -t BASE < <(PYTHONPATH=. $M/.venv/bin/python scripts/data_free_queue/recipe_from_run.py "$M/runs/data_free_latent/$REFERENCE")
  if [ "${#BASE[@]}" -lt 10 ]; then echo "could not read the settings of $REFERENCE; $NAME not started"; rmdir "$O" 2> /dev/null; exit 1; fi
else
  BASE=(--set accumulate=4 --set lr=6.364e-4 --set lr_warmup_steps=125 --set lr_floor_fraction=1.0
    --set mult_rate=0.08 --set mult_rates.bridge=0.2 --set mult_rates.delete=0.2 --set mult_rates.fill=0.2
    --set mult_rates.widen=0.2 --set mult_rates.thicken=0.2 --set mult_rates.barrier=0.2
    --set diag_every=250 --set ckpt_every=100 --set log_every=25 --set diag_resolution=96
    --set clusters=4 --set codes_per_cluster=8 --set global_codes=True --set diversity_mode=nearest
    --set spread_weight=0.0 --set batch_neighbours=True --set pool_size=0 --set pool_points=1024 --set nearest_k=8
    --set anchor_path=$ANCHORS --set anchor_codes_per_pass=4
    --set anchor_weight=100.0 --set anchor_surface_weight=1.0 --set anchor_embed_lr=3e-4
    --set anchor_freeze_steps=500 --set anchor_ramp_steps=1000
    --set anchor_radius=0.5 --set anchor_exempt_field=True)
fi
bash "$W/scripts/data_free_queue/run_guarded.sh" "$O" "$W" "train_data_free_latent.py --output-dir $O " -- --steps "$STEPS" --device cuda --seed "$SEED" "${BASE[@]}" "$@"
echo "$NAME finished: $(grep -E '^diag' "$O/train.log" | tail -n 1 | cut -c1-260)"
