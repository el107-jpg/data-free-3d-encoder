#!/usr/bin/env python3
"""Run a training entry point with caps on PyTorch GPU memory and GPU busy time.

Keeps headroom on the GPU for the desktop and other apps when the training GPU
also drives the display.

* --gpu-fraction F  caps the PyTorch allocator at F of total GPU memory, so
  training hits a CUDA OOM in its own process instead of starving other apps.
* --gpu-util U      duty-cycles training: after every optimizer.step() it waits
  for the GPU to finish, then sleeps so that training keeps the GPU busy at
  most ~U of wall-clock time. The idle gaps let the compositor and apps render.
  1.0 disables throttling.

Lives outside ``arm_planner`` on purpose: editing ``arm_planner`` would change
the geometry pool-cache signature.

Usage:
    python scripts/gpu_capped.py [--gpu-fraction 0.75] [--gpu-util 0.7] -m arm_planner.object_geometry.full_library ARGS...
    python scripts/gpu_capped.py --gpu-util 0.6 scripts/train_latent_regularized.py ARGS...

Defaults come from the GPU_MEM_FRACTION and GPU_UTIL env vars (0.75 and 1.0).
"""
from __future__ import annotations

import os
import runpy
import sys
import time


def _pop_flag(argv: list[str], name: str, default: float) -> float:
    """Consume leading ``--name V`` / ``--name=V`` options (before the target)."""
    value = default
    i = 0
    while i < len(argv) and argv[i].startswith("--gpu-"):
        if argv[i] == name:
            value = float(argv[i + 1])
            del argv[i:i + 2]
        elif argv[i].startswith(name + "="):
            value = float(argv[i].split("=", 1)[1])
            del argv[i]
        else:
            i += 1
    return value


def _install_util_throttle(target: float) -> None:
    import torch
    from torch.optim.optimizer import register_optimizer_step_post_hook

    state = {"since": time.perf_counter()}
    ratio = (1.0 - target) / target

    def hook(optimizer, args, kwargs):
        torch.cuda.synchronize()
        now = time.perf_counter()
        busy = now - state["since"]
        time.sleep(min(busy * ratio, 1.0))
        state["since"] = time.perf_counter()

    register_optimizer_step_post_hook(hook)


def main() -> None:
    argv = sys.argv[1:]
    fraction = _pop_flag(argv, "--gpu-fraction", float(os.environ.get("GPU_MEM_FRACTION", "0.75")))
    util = _pop_flag(argv, "--gpu-util", float(os.environ.get("GPU_UTIL", "1.0")))
    if not 0.0 < fraction <= 1.0:
        sys.exit(f"--gpu-fraction must be in (0, 1], got {fraction}")
    if not 0.05 <= util <= 1.0:
        sys.exit(f"--gpu-util must be in [0.05, 1], got {util}")
    if not argv:
        sys.exit(__doc__)

    # Return freed blocks to the driver more readily and avoid fragmentation.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_per_process_memory_fraction(fraction, i)
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        msg = f"memory capped at {fraction:.0%} (~{fraction * total:.1f} of {total:.1f} GiB)"
        if util < 1.0:
            _install_util_throttle(util)
            msg += f"; busy time throttled to ~{util:.0%}"
        print(f"[gpu_capped] {msg}", file=sys.stderr)

    if argv[0] == "-m":
        module = argv[1]
        sys.argv = [module] + argv[2:]
        runpy.run_module(module, run_name="__main__", alter_sys=True)
    else:
        script = argv[0]
        sys.argv = argv
        sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
        runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
