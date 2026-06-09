# -*- coding: utf-8 -*-
"""
System overview figure (Fig. 1): the proposed gated closed-loop subspace--RFS
framework.  Rendered for a DOUBLE-COLUMN (figure*) placement so the text stays
large and uncropped.  Output: fig_system.png.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
BLUE, GREEN, RED, GRAY = "#1f77b4", "#2ca02c", "#d62728", "#444444"


def box(ax, x, y, w, h, text, ec, fc="white", fs=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.010,rounding_size=0.015",
                                linewidth=2.0, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, zorder=3, clip_on=False)


def arrow(ax, p0, p1, color=GRAY, lw=2.2):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=20,
                                 linewidth=lw, color=color, zorder=1))


def tag(ax, x, y, text, color, fs=10):
    ax.text(x, y, text, ha="center", va="top", fontsize=fs, color=color,
            style="italic", clip_on=False)


fig, ax = plt.subplots(figsize=(7.4, 2.15))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

yt, h = 0.58, 0.30        # main row
# --- main pipeline (left -> right) ---
box(ax, 0.005, yt, 0.135, h, "Array\nsnapshots $\\mathbf{X}_n$", GRAY, fs=10)
box(ax, 0.190, yt, 0.255, h,
    "Subspace front-end\nCOP (4th-order)  /\nMUSIC (2nd-order)", BLUE, fc="#eaf2fb", fs=10)
box(ax, 0.525, yt, 0.285, h,
    "Interchangeable RFS tracker\nSOTA: TO-PHD / LMB / $\\delta$-GLMB\nphysics CV/CA motion", GREEN, fc="#eafaef", fs=10)
box(ax, 0.875, yt, 0.120, h, "Labeled\ntracks", GRAY, fs=10)

arrow(ax, (0.140, yt + h / 2), (0.190, yt + h / 2))
arrow(ax, (0.445, yt + h / 2), (0.525, yt + h / 2))
ax.text(0.485, yt + h / 2 + 0.085, "DOA est.\n$\\hat{\\theta}$",
        ha="center", va="bottom", fontsize=9.5, color=GRAY, clip_on=False)
arrow(ax, (0.810, yt + h / 2), (0.875, yt + h / 2))

tag(ax, 0.3175, yt - 0.035, "front-end-agnostic", BLUE)
tag(ax, 0.6675, yt - 0.035, "back-end-agnostic", GREEN)

# --- feedback path (tracker -> gate/refinement -> front-end) ---
yb, hb = 0.06, 0.22
box(ax, 0.15, yb, 0.64, hb,
    "Gated T-COP feedback (closed loop)\n"
    "prediction $\\mathbf{F}\\mathbf{m}$ $\\to$ velocity + subspace-bias gate "
    "$\\to$ Grassmann fusion", RED, fc="#fdeaea", fs=9)
arrow(ax, (0.6675, yt), (0.6675, yb + hb), color=RED)      # tracker -> down
arrow(ax, (0.3175, yb + hb), (0.3175, yt), color=RED)      # up -> front-end
tag(ax, 0.4425, yb - 0.02, "Theorems 1, 2  (minimum-variance, no-harm gate)", RED, fs=10)

fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.06)
out = os.path.join(HERE, "fig_system.png")
fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)
print("saved", out)
