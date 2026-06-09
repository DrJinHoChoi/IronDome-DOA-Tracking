# -*- coding: utf-8 -*-
"""
System overview figure (Fig. 1): the proposed gated closed-loop subspace--RFS
framework. Front-end-agnostic estimator -> RFS tracker (back-end-agnostic, physics
motion) -> tracks; the tracker prediction is fed back through a validation gate and
a T-COP subspace refinement, closing the loop. Output: fig_system.png.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
BLUE, GREEN, RED, GRAY = "#1f77b4", "#2ca02c", "#d62728", "#555555"


def box(ax, x, y, w, h, text, ec, fc="white", fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=1.6, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=3)


def arrow(ax, p0, p1, color=GRAY, lw=1.8, style="-|>", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=14,
                                 linewidth=lw, color=color, zorder=1,
                                 connectionstyle=f"arc3,rad={rad}", linestyle=ls))


fig, ax = plt.subplots(figsize=(7.2, 2.9))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

yt = 0.60  # main row
h = 0.26
# --- main pipeline (left -> right) ---
box(ax, 0.015, yt, 0.135, h, "Array\nsnapshots\n$\\mathbf{X}_n$", GRAY)
box(ax, 0.205, yt, 0.255, h,
    "Subspace front-end\nCOP (4th-order) /\nMUSIC (2nd-order)", BLUE, fc="#eaf2fb")
box(ax, 0.545, yt, 0.275, h,
    "RFS tracker\nTO-PHD / LMB / $\\delta$-GLMB\nphysics CV/CA motion", GREEN, fc="#eafaef")
box(ax, 0.865, yt, 0.125, h, "Labeled\ntracks", GRAY)

arrow(ax, (0.150, yt + h / 2), (0.205, yt + h / 2))
arrow(ax, (0.460, yt + h / 2), (0.545, yt + h / 2))
ax.text(0.5025, yt + h / 2 + 0.075, "DOA est.\n$\\hat{\\theta}$", ha="center", va="bottom", fontsize=8, color=GRAY)
arrow(ax, (0.820, yt + h / 2), (0.865, yt + h / 2))

# --- agnostic tags ---
ax.text(0.3325, yt - 0.055, "front-end-agnostic", ha="center", va="top", fontsize=8, color=BLUE, style="italic")
ax.text(0.6825, yt - 0.055, "back-end-agnostic", ha="center", va="top", fontsize=8, color=GREEN, style="italic")

# --- feedback path (tracker -> gate/refinement -> front-end) ---
yb = 0.10
box(ax, 0.235, yb, 0.50, 0.20,
    "Gated T-COP refinement:  CV/CA prediction $\\mathbf{F}\\mathbf{m}$\n"
    "$\\to$ velocity and subspace-bias gate $\\to$ Grassmann fusion", RED, fc="#fdeaea", fs=8.5)
# tracker down into feedback box
arrow(ax, (0.6825, yt), (0.6825, yb + 0.20), color=RED, rad=0.0)
# feedback box up into front-end (close the loop)
arrow(ax, (0.30, yb + 0.20), (0.30, yt), color=RED, rad=0.0)
ax.text(0.485, yb - 0.045, "closed loop (Theorems 1, 2; no-harm gate)",
        ha="center", va="top", fontsize=8, color=RED, style="italic")

fig.tight_layout(pad=0.3)
out = os.path.join(HERE, "fig_system.png")
fig.savefig(out, dpi=160, bbox_inches="tight")
plt.close(fig)
print("saved", out)
