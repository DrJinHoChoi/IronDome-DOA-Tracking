"""Generate architecture figure for Mamba-COP-RL paper."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

fig, ax = plt.subplots(figsize=(9, 3.8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 4)
ax.axis('off')

# ── Colors ────────────────────────────────────────────────
C_ENV   = '#2196F3'   # blue  — environment
C_SSM   = '#4CAF50'   # green — SSM encoder
C_AC    = '#FF9800'   # orange — actor-critic
C_PHD   = '#9C27B0'   # purple — GM-PHD
C_ARROW = '#424242'
C_BG    = '#F5F5F5'

def box(ax, x, y, w, h, label, sublabel='', color='#607D8B', fs=9, sfs=7.5):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.08",
                          facecolor=color, edgecolor='white',
                          linewidth=1.5, alpha=0.92, zorder=3)
    ax.add_patch(rect)
    cy = y + h/2 + (0.12 if sublabel else 0)
    ax.text(x + w/2, cy, label, ha='center', va='center',
            fontsize=fs, fontweight='bold', color='white', zorder=4)
    if sublabel:
        ax.text(x + w/2, y + h/2 - 0.18, sublabel, ha='center', va='center',
                fontsize=sfs, color='#E0E0E0', zorder=4)

def arrow(ax, x0, y0, x1, y1, label='', color=C_ARROW, lw=1.5):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, connectionstyle='arc3,rad=0'))
    if label:
        mx, my = (x0+x1)/2, (y0+y1)/2
        ax.text(mx, my+0.18, label, ha='center', va='bottom',
                fontsize=7, color='#616161',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1))

def darrow(ax, x0, y0, x1, y1, label='', color=C_ARROW, lw=1.5, rad=0.0):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                connectionstyle=f'arc3,rad={rad}'))
    if label:
        mx, my = (x0+x1)/2, (y0+y1)/2+0.15
        ax.text(mx, my, label, ha='center', va='bottom',
                fontsize=7, color='#616161',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1))

# ── Background panel ─────────────────────────────────────
bg = FancyBboxPatch((0.1, 0.15), 9.8, 3.6,
                    boxstyle="round,pad=0.1",
                    facecolor=C_BG, edgecolor='#BDBDBD',
                    linewidth=1, alpha=0.5, zorder=1)
ax.add_patch(bg)

# ── Title ─────────────────────────────────────────────────
ax.text(5, 3.8, 'Mamba-COP-RL Pipeline',
        ha='center', va='center', fontsize=11,
        fontweight='bold', color='#212121')

# ── Blocks ────────────────────────────────────────────────
# 1. Sensor Array
box(ax, 0.2, 1.5, 1.3, 1.0, 'Array', '$M{=}8$ sensors', C_ENV, fs=8.5)

# 2. COP Spectrum
box(ax, 1.9, 1.5, 1.5, 1.0, 'COP', r'$P_{\rm COP}(\theta)$', C_ENV, fs=8.5)

# 3. SSM Encoder
box(ax, 3.8, 1.2, 1.8, 1.6, 'SSM Encoder',
    r'$d_h{=}48,\;d_s{=}24$', C_SSM, fs=8.5)

# 4. Actor-Critic
box(ax, 6.0, 1.5, 1.7, 1.0, 'Actor-Critic',
    '64→64, Tanh', C_AC, fs=8.5)

# 5. GM-PHD Filter
box(ax, 8.1, 1.5, 1.7, 1.0, 'GM-PHD', 'Filter', C_PHD, fs=8.5)

# ── Arrows (main flow) ────────────────────────────────────
arrow(ax, 1.5, 2.0, 1.9, 2.0, r'$X\in\mathbb{C}^{M\times T}$')
arrow(ax, 3.4, 2.0, 3.8, 2.0, r'$\tilde{P}_{\rm COP}\in\mathbb{R}^{181}$')
arrow(ax, 5.6, 2.0, 6.0, 2.0, r'$\tilde{\mathbf{o}}_t\in\mathbb{R}^{183}$')
arrow(ax, 7.7, 2.0, 8.1, 2.0, r'$(w_b,\tau_p,p_D)$')

# ── Hidden state arrow (SSM self-loop) ────────────────────
ax.annotate('', xy=(3.8, 1.5), xytext=(5.6, 1.5),
            arrowprops=dict(arrowstyle='<->', color=C_SSM, lw=1.5,
                            connectionstyle='arc3,rad=0.55'))
ax.text(4.7, 0.75, r'$\mathbf{h}_t \in \mathbb{R}^{d_s}$  (hidden state)',
        ha='center', va='center', fontsize=7.5,
        color=C_SSM, style='italic')

# ── GM-PHD → COP feedback arrow ──────────────────────────
ax.annotate('', xy=(2.65, 1.5), xytext=(8.95, 1.5),
            arrowprops=dict(arrowstyle='->',
                            color=C_PHD, lw=1.5,
                            connectionstyle='arc3,rad=-0.55'))
ax.text(5.8, 0.35, r'Predicted DOAs (tracker feedback)',
        ha='center', va='center', fontsize=7.5,
        color=C_PHD, style='italic')

# ── Observations label above COP ──────────────────────────
ax.text(2.65, 2.75, r'$\mathbf{o}_t = [\tilde{P}_{\rm COP},'
        r'\;t/T,\;\rho_{\rm SNR}]^\top$',
        ha='center', va='center', fontsize=7.5, color='#37474F',
        bbox=dict(facecolor='white', edgecolor='#CFD8DC',
                  boxstyle='round,pad=0.25', alpha=0.9))

# ── Inference mode note ───────────────────────────────────
ax.text(4.7, 3.55,
        r'Inference: SSM.step() $\;O(d_s)$/scan $\;\Rightarrow$ real-time Edge',
        ha='center', va='center', fontsize=7.5, color='#388E3C',
        bbox=dict(facecolor='#E8F5E9', edgecolor='#A5D6A7',
                  boxstyle='round,pad=0.3', alpha=0.95))

# ── BPTT note ─────────────────────────────────────────────
ax.text(4.7, 3.1,
        r'Training: SSM.forward($\mathbf{o}_{1:T}$) $\Rightarrow$ BPTT through encoder',
        ha='center', va='center', fontsize=7.5, color='#E65100',
        bbox=dict(facecolor='#FFF3E0', edgecolor='#FFCC80',
                  boxstyle='round,pad=0.3', alpha=0.95))

# ── Legend ────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(color=C_ENV,  label='Environment / COP'),
    mpatches.Patch(color=C_SSM,  label='SSM Encoder (Mamba)'),
    mpatches.Patch(color=C_AC,   label='Actor-Critic (PPO)'),
    mpatches.Patch(color=C_PHD,  label='GM-PHD Filter'),
]
ax.legend(handles=legend_items, loc='lower right',
          fontsize=7, framealpha=0.85, edgecolor='#BDBDBD',
          bbox_to_anchor=(0.99, 0.02))

plt.tight_layout(pad=0.2)
out = 'fig_architecture.pdf'
plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
plt.savefig(out.replace('.pdf', '.png'), dpi=200,
            bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {out}")
