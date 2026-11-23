"""Chart of layer_inject.py: per hidden layer, code directions injected and the share of the surface's response.

usage: plot_layer_inject.py <layer_inject.json> <out.png> <label> [<label> ...]   (2 to 4 labels)
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

data = json.load(open(sys.argv[1]))
labels = sys.argv[3:]
INK, MUTED, GRID, PAPER = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COLOURS = ["#2a78d6", "#0e3a78", "#eb6834", "#a8401a"]
L = 8
layers = np.arange(1, L + 1)
fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
fig.patch.set_facecolor(PAPER)
width = 0.8 / len(labels)
for i, (label, colour) in enumerate(zip(labels, COLOURS)):
    r = data[label]
    inj = [r[f"{k}|injected"]["effective"] for k in range(L)]
    at = [r[f"{k}|at layer"]["effective"] for k in range(L)]
    sizes = np.array([r[f"{k}|surface"]["size"] for k in range(L)])
    share = sizes / sizes.sum()
    name = label.replace("no base", "no base, update").replace("J4", "J4, update").replace("J7", "J7, update")
    axes[0].plot(layers, inj, "-o", color=colour, lw=2, ms=4, label=name)
    axes[1].plot(layers, at, "-o", color=colour, lw=2, ms=4, label=name)
    axes[2].bar(layers - 0.4 + width * (i + 0.5), share, width=width * 0.92, color=colour, label=name)
titles = ["Code directions in each layer's scale and shift\n(what the side network injects)",
          "Code directions in each layer's output\n(when only that layer's scale and shift move)",
          "Share of the surface's response to the code\ncoming from each layer's injection"]
for ax, title in zip(axes, titles):
    ax.set_title(title, fontsize=10.5, color=INK, loc="left")
    ax.set_xticks(layers)
    ax.set_xticklabels([f"main {k}" for k in layers], fontsize=8.5, color=MUTED)
    ax.set_facecolor(PAPER)
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
    for side in ax.spines.values():
        side.set_visible(False)
axes[0].set_ylabel("effective number of directions (of 63)", fontsize=9.5, color=MUTED)
axes[1].set_ylabel("effective number of directions (of 63)", fontsize=9.5, color=MUTED)
axes[2].set_ylabel("share of the surface motion", fontsize=9.5, color=MUTED)
axes[0].set_ylim(0, 63); axes[1].set_ylim(0, 63); axes[2].set_ylim(0, 1)
axes[2].legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper right")
fig.suptitle("Where the code enters the main network and where it reaches the shape (6 random codes; surface motion with pose removed)",
             fontsize=11.5, color=INK, x=0.01, ha="left")
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=115, facecolor=PAPER)
print("saved", sys.argv[2])
