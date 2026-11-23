"""Distil a trained data-free decoder into a changed decoder form and save a checkpoint training can continue from.

usage: distill_data_free.py --teacher <checkpoint> --output <checkpoint to write> [--steps N] [--lr LR]
       [--device cuda] [--set key=value ...]

The student is the teacher's checkpoint loaded with the --set changes (e.g. sphere_base=False
outer_limit=min), so it starts from the teacher's weights, anchor codes, multipliers and update count. It is
trained to reproduce the teacher's field (value and slope) at random codes and the anchor codes, its
optimiser state is then cleared, and the full trainer state is written to --output. Continue it with
cont_run.sh (or the trainer's --resume after copying it to checkpoint_last.pt in a new run folder).
A JSON log is written next to --output.
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arm_planner.data_free.distill import agreement, distill  # noqa: E402
from arm_planner.data_free.train import Trainer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--set", action="append", default=[], dest="overrides")
    args = ap.parse_args()
    if os.path.exists(args.output):
        raise SystemExit(f"{args.output} exists; refusing to overwrite")
    teacher = Trainer.from_checkpoint(args.teacher, device=args.device, tensorboard=False)
    student = Trainer.from_checkpoint(args.teacher, device=args.device, tensorboard=False, overrides=args.overrides)
    dim = int(student.cfg.latent_dim)
    before = agreement(student.model, teacher.model, dim)
    print("before:", json.dumps(before), flush=True)
    history = distill(student.model, teacher.model, dim, args.steps, anchor_codes=student.anchor_codes, lr=args.lr,
                      generator=torch.Generator(device=args.device).manual_seed(int(student.cfg.seed) + 17),
                      log=lambda r: print(json.dumps(r), flush=True))
    after = agreement(student.model, teacher.model, dim)
    print("after:", json.dumps(after), flush=True)
    student.optimizer.state.clear()                       # the function changed: start Adam afresh
    student.model.train()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    student.save_checkpoint(args.output)
    with open(os.path.splitext(args.output)[0] + "_distill.json", "w") as handle:
        json.dump({"teacher": args.teacher, "overrides": args.overrides, "steps": args.steps, "lr": args.lr,
                   "before": before, "after": after, "history": history}, handle, indent=1)
    print("saved", args.output, flush=True)


if __name__ == "__main__":
    main()
