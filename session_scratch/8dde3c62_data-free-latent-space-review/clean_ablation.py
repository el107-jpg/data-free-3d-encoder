"""Resume the pre-clean model (run d, step 10000) under three action-loss settings.

Shows what a clean-up does to the shapes and which action loss is responsible.
Saves one z=0 filmstrip per setting plus a JSON of diagnostics and weighted loss terms.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from arm_planner.data_free.train import Trainer

CKPT = "/home/eric-luo/.claude/jobs/8dde3c62/tmp/df_train/d/checkpoint_last.pt"
OUT = "/home/eric-luo/.claude/jobs/8dde3c62/tmp/clean_ablation"
IMG = "/home/eric-luo/Downloads"
os.makedirs(OUT, exist_ok=True)

STEPS, EVERY, PROBES, RES = 900, 150, 8, 224
COMMON = ["steps=12000", "lr_floor_fraction=0.17"]          # learning rate about 5e-5, as during run e's episodes
CONDITIONS = {
    "all_actions": [],
    "no_thicken": ["mult_ranges.thicken=1,1"],
    "fixed_graph": [],
    "thicken_only": ["mult_ranges.delete=1,1", "mult_ranges.widen=1,1",
                     "mult_ranges.bridge=1,1", "mult_ranges.fill=1,1"],
}
NAMES = ["eikonal", "consistency", "barrier", "pose_centroid", "pose_scale", "pose_axes",
         "bridge", "delete", "fill", "widen", "thicken"]


def slices(trainer):
    axis = torch.linspace(-1, 1, RES, device="cuda")
    ga, gb = torch.meshgrid(axis, axis, indexing="xy")
    pts = torch.stack([ga, gb, torch.zeros_like(ga)], dim=-1).reshape(1, -1, 3)
    codes = trainer.bank.codes[:PROBES]
    trainer.model.eval()
    with torch.no_grad():
        f = trainer.model(codes, pts.expand(PROBES, -1, -1)).reshape(PROBES, RES, RES).cpu().numpy()
    trainer.model.train()
    return f


def run(name, overrides):
    torch.manual_seed(0)
    trainer = Trainer.from_checkpoint(CKPT, device="cuda", output_dir=None,
                                      overrides=COMMON + overrides, tensorboard=False)
    start = trainer.step_count
    frames, log, window = [], [], []
    for i in range(STEPS + 1):
        if i % EVERY == 0:
            diag = {k: (float(v) if isinstance(v, (int, float)) else v)
                    for k, v in trainer.run_diagnostics().items() if isinstance(v, (int, float))}
            mean = {}
            if window:
                for n in NAMES:
                    mean["w_" + n] = sum(m["mult/" + n] * m["term/" + n] for m in window) / len(window)
                    mean["mult_" + n] = window[-1]["mult/" + n]
                for k in ["diversity", "div/mean_ratio", "graph/wall_sites", "graph/gap_sites",
                          "graph/fragments_kind0", "grad_norm", "term/eikonal"]:
                    mean[k] = sum(m[k] for m in window) / len(window)
            frames.append((start + i, slices(trainer)))
            log.append({"step": start + i, "diag": diag, "train": mean})
            print(name, start + i, {k: round(v, 3) for k, v in diag.items()
                                    if any(s in k for s in ("piece", "handle", "live", "pose_share", "wobble"))},
                  {k: round(v, 3) for k, v in mean.items() if k.startswith("w_") and abs(v) > 1e-3},
                  flush=True)
            window = []
        if i < STEPS:
            window.append(trainer.step())
    json.dump(log, open(f"{OUT}/{name}.json", "w"), indent=1)

    a = np.linspace(-1, 1, RES)
    theta = np.linspace(0, 2 * np.pi, 200)
    fig, axes = plt.subplots(len(frames), PROBES, figsize=(2.2 * PROBES, 2.25 * len(frames)))
    for r, (step, f) in enumerate(frames):
        for c in range(PROBES):
            ax = axes[r, c]
            ax.contourf(a, a, f[c], levels=[0, 10], colors=["#9ecae1"])
            ax.contour(a, a, f[c], levels=[0], colors="#08519c", linewidths=1.6)
            ax.plot(0.5 * np.cos(theta), 0.5 * np.sin(theta), color="#d95f02", lw=0.7, ls="--")
            ax.set_aspect("equal"); ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"probe code {c}", fontsize=9)
            if c == 0:
                ax.set_ylabel(f"+{step - start} steps", fontsize=10)
    fig.suptitle(f"z = 0 slice of 8 probe codes while training continues from the pre-clean model "
                 f"(run d, step {start}); setting: {name}\n"
                 "Blue: shape and its surface; dashed orange: initial sphere R = 0.5", fontsize=11)
    fig.tight_layout()
    path = f"{IMG}/data_free_clean_{name}.png"
    fig.savefig(path, dpi=95)
    plt.close(fig)
    print("saved", path, flush=True)
    trainer.save_checkpoint(f"{OUT}/{name}_checkpoint.pt")
    del trainer
    torch.cuda.empty_cache()


only = sys.argv[1:] or list(CONDITIONS)
for cond in only:
    run(cond, CONDITIONS[cond])
