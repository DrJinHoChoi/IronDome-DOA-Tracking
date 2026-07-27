# -*- coding: utf-8 -*-
"""
System overview figure (Fig. 1): the proposed gated closed-loop subspace--RFS
framework. VERTICAL layout so it fits a SINGLE IEEE column with readable text.
Output: fig_system.png.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
BLUE, GREEN, RED, GRAY = "#1f77b4", "#2ca02c", "#d62728", "#444444"
CX = 0.57   # main-column center


def box(ax, cy, w, h, text, ec, fc="white", fs=9.0):
    x, y = CX - w / 2, cy - h / 2
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.008,rounding_size=0.02",
                                linewidth=1.8, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(CX, cy, text, ha="center", va="center", fontsize=fs, zorder=3, clip_on=False)


def down(ax, y0, y1, label=None):
    ax.add_patch(FancyArrowPatch((CX, y0), (CX, y1), arrowstyle="-|>",
                                 mutation_scale=14, linewidth=1.8, color=GRAY, zorder=1))
    if label:
        ax.text(CX + 0.05, (y0 + y1) / 2, label, ha="left", va="center",
                fontsize=8.5, color=GRAY, clip_on=False)


fig, ax = plt.subplots(figsize=(3.45, 4.1))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# --- vertical pipeline (top -> bottom) ---
box(ax, 0.935, 0.46, 0.075, "Array snapshots $\\mathbf{X}_n$", GRAY)
box(ax, 0.74, 0.70, 0.155,
    "Subspace front-end  (front-end-agnostic)\nCOP (4th-order) / MUSIC (2nd-order)", BLUE, fc="#eaf2fb")
box(ax, 0.45, 0.70, 0.175,
    "RFS tracker  (back-end-agnostic)\n$\\delta$-GLMB / LMB / TO-PHD\nphysics CV/CA motion", GREEN, fc="#eafaef")
box(ax, 0.18, 0.46, 0.075, "Labeled tracks", GRAY)

down(ax, 0.8975, 0.8175)                       # X -> front-end
down(ax, 0.6625, 0.5375, "$\\hat{\\theta}$")    # front-end -> tracker
down(ax, 0.3625, 0.2175)                        # tracker -> tracks

# --- feedback loop (tracker -> gate/refinement -> front-end), left side ---
ax.add_patch(FancyArrowPatch((CX - 0.35, 0.45), (CX - 0.35, 0.74),
                             arrowstyle="-|>", mutation_scale=14, linewidth=1.8,
                             color=RED, zorder=1, connectionstyle="arc3,rad=-0.55"))
ax.text(0.045, 0.595, "gated T-COP feedback\nCV/CA prediction $\\to$ gate\n$\\to$ Grassmann fusion\n(Theorems 1, 2)",
        ha="left", va="center", fontsize=8.0, color=RED, style="italic", clip_on=False, rotation=90)

fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
out = os.path.join(HERE, "fig_system.png")
fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.04)
plt.close(fig)
print("saved", out)
