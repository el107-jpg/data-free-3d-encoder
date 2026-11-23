"""Chart of layer_dims.py results: independent code directions at each stage, per checkpoint.

usage: plot_layer_dims.py <layer_dims.json> <out.png>
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

data = json.load(open(sys.argv[1]))
STAGES = [("side network, layer 1", "side 1"), ("side network, layer 2", "side 2"),
          ("scale and shift, all layers", "scale &\nshift"),
          ] + [(f"main network, hidden layer {k}", f"main {k}") for k in range(1, 9)] + [
          ("field at points in the ball", "field"), ("surface motion (pose removed)", "surface")]
INK, MUTED, GRID, PAPER = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUES = ["#c6dbf7", "#9cc0ee", "#6fa3e3", "#4a87d6", "#2a6bc4", "#1a52a3", "#0e3a78"]
J4C = ["#f3b48f", "#eb6834", "#a8401a"]
xs = np.arange(len(STAGES))
fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.6), sharey=True)
fig.patch.set_facecolor(PAPER)
no_base = [(k, v) for k, v in data.items() if k.startswith("no base")]
j4 = [(k, v) for k, v in data.items() if k.startswith("J4")]
for ax, groups, title in ((axes[0], [(no_base, BLUES, "-")], "Without the fixed sphere (S5 to 2,500, then S6), over training"),
                          (axes[1], [(no_base[-1:], [BLUES[-1]], "-"), (j4, J4C, "--")], "With the fixed sphere (J4, dashed) against without it at 5,000")):
    for runs, colours, style in groups:
        for (label, v), colour in zip(runs, colours[-len(runs):] if len(runs) < len(colours) else colours):
            ys = [v[s]["effective"] for s, _ in STAGES]
            name = label.replace("no base", "no base, update").replace("J4", "J4, update")
            ax.plot(xs, ys, style, color=colour, lw=2, marker="o", ms=4, label=name)
            ax.text(xs[-1] + 0.15, ys[-1], f"{ys[-1]:.1f}", fontsize=8, color=INK, va="center")
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 3, 5, 10, 20, 30, 63])
    ax.set_yticklabels(["1", "2", "3", "5", "10", "20", "30", "63"])
    ax.set_ylim(1, 70)
    ax.set_xticks(xs)
    ax.set_xticklabels([s for _, s in STAGES], fontsize=8.5, color=MUTED)
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    ax.axvspan(2.5, 10.5, color="#f1f0ec", zorder=0)
    ax.text(6.5, 55, "main network", fontsize=9, color=MUTED, ha="center")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="lower left")
    ax.set_facecolor(PAPER)
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
    for side in ax.spines.values():
        side.set_visible(False)
axes[0].set_ylabel("effective number of code directions (of 63)", fontsize=9.5, color=MUTED)
fig.suptitle("Where the code's 63 directions collapse: effective dimension at each stage of the decoder "
             "(8 random codes; field and surface use the shape's own surface)", fontsize=11.5, color=INK, x=0.01, ha="left")
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=115, facecolor=PAPER)
print("saved", sys.argv[2])
