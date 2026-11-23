#!/bin/bash
cd /home/eric-luo/Downloads/pointcloud-motion-planning
export PYTHONPATH=$PWD
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader > /home/eric-luo/.claude/jobs/8dde3c62/tmp/fd_bench/smi_before.txt
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader >> /home/eric-luo/.claude/jobs/8dde3c62/tmp/fd_bench/smi_before.txt
.venv/bin/python /home/eric-luo/.claude/jobs/8dde3c62/tmp/fd_bench/fd_bench.py --out /home/eric-luo/.claude/jobs/8dde3c62/tmp/fd_bench/random_final.json --sections acc,pgrad,time --sizes 512:32 > /home/eric-luo/.claude/jobs/8dde3c62/tmp/fd_bench/random_final.log 2>&1
.venv/bin/python /home/eric-luo/.claude/jobs/8dde3c62/tmp/fd_bench/fd_bench.py --out /home/eric-luo/.claude/jobs/8dde3c62/tmp/fd_bench/trained_final.json --run runs/object_geometry_latent_set16_unpack_d32_phase3nowire_cos3e4_50k_20261105_v1_r4 --codes /home/eric-luo/.claude/jobs/8dde3c62/tmp/fd_bench/codes_d32.pt --sections acc,pgrad,time,amp > /home/eric-luo/.claude/jobs/8dde3c62/tmp/fd_bench/trained_final.log 2>&1
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader > /home/eric-luo/.claude/jobs/8dde3c62/tmp/fd_bench/smi_after.txt
echo done
