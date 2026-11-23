#!/usr/bin/env bash
# Bring this folder up to date with the sources on the ORIGINAL machine (it reads fixed paths there, so it only
# works on that machine). Copy-only and incremental; checkpoints are never copied. Finishes with a disk-level
# comparison against the sources and a fresh SHA256SUMS.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=/home/eric-luo/Downloads/pointcloud-motion-planning
bash "$HERE/build_transfer.sh"
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 "$ROOT/.venv/bin/python" "$HERE/extract_configs.py" "$ROOT/data_free_latent_transfer/code/rounded-box-base"
python3 "$HERE/run_index.py"
python3 "$HERE/verify_transfer.py" verify
python3 "$HERE/verify_transfer.py" sums
