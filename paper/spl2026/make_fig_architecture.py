"""Architecture figure — Mamba-COP-RL (SPL 2026, enlarged fonts).
Single-row pipeline with uniform box sizes and equal spacing.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.patches import FancyArrowPatch
import numpy as np

# Canvas (slightly taller to fit larger fonts)
W, H = 14.0, 4.2
fig = plt.figure(figsize=(W, H))
ax  = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, H)
ax.axis('off')
ax.set_facecolor('white')

C_ENV  = '#2E86C1'
C_SSM  = '#1E8449'
C_AC   = '#CA6F1E'
C_PHD  = '#7D3C98'
C_TEXT = '#1A1A1A'
C_ANN  = '#444444'

N      = 5
BW     = 2.00
BH     = 1.35
GAP    = 0.50
TOTAL  = N*BW + (N-1)*GAP
MARGIN = (W - TOTAL) / 2

BY = (H - BH) / 2 - 0.15

xs = [MARGIN + i*(BW+GAP) for i in range(N)]

LABELS = [
    ('Sensor\nArray',    r'$M{=}8$ ULA',             C_ENV),
    ('COP\nSpectrum',    r'$N_\theta{=}181$',         C_ENV),
    ('SSM\nEncoder',     r'$d_h{=}48,\ d_s{=}24$',   C_SSM),
    ('Actor-Critic',     r'MLP $183{\to}64{\to}3$',   C_AC),
    ('GM-PHD\nFilter',   r'$(w_b,\tau_p,p_D)$',       C_PHD),
]

for x, (title, sub, color) in zip(xs, LABELS):
    rect = FancyBboxPatch((x, BY), BW, BH,
                          boxstyle='round,pad=0.10',
                          facecolor=color, edgecolor='white',
                          linewidth=2.0, alpha=0.90, zorder=3)
    ax.add_patch(rect)
    ax.text(x + BW/2, BY + BH*0.62, title,
            ha='center', va='center', fontsize=17,
            fontweight='bold', color='white', zorder=4,
            linespacing=1.3)
    ax.text(x + BW/2, BY + BH*0.22, sub,
            ha='center', va='center', fontsize=14,
            color='#E8E8E8', zorder=4)

ARROW_LABELS = [
    r'$\mathbf{X}{\in}\mathbb{C}^{M{\times}L}$',
    r'$\mathbf{o}_t{\in}\mathbb{R}^{183}$',
    r'$\tilde{\mathbf{o}}_t{\in}\mathbb{R}^{183}$',
    r'$\mathbf{a}_t{\in}[-1,1]^3$',
]
CY = BY + BH/2

for i, lbl in enumerate(ARROW_LABELS):
    x0 = xs[i] + BW
    x1 = xs[i+1]
    ax.annotate('', xy=(x1, CY), xytext=(x0, CY),
                arrowprops=dict(arrowstyle='->', color=C_ANN,
                                lw=1.8, mutation_scale=16))
    ax.text((x0+x1)/2, CY + 0.26, lbl,
            ha='center', va='bottom', fontsize=13,
            color=C_ANN,
            bbox=dict(facecolor='white', edgecolor='none',
                      alpha=0.85, pad=1))

sx = xs[2]; sw = BW
ax.annotate('',
            xy=(sx, BY + 0.28),
            xytext=(sx + sw, BY + 0.28),
            arrowprops=dict(arrowstyle='<->',
                            color=C_SSM, lw=1.6,
                            connectionstyle='arc3,rad=0.55'))
ax.text(sx + sw/2, BY - 0.32,
        r'$\mathbf{h}_t{\in}\mathbb{R}^{d_s}$  hidden state',
        ha='center', va='center', fontsize=14,
        color=C_SSM, style='italic')

x_start = xs[4] + BW/2
x_end   = xs[1] + BW/2
ax.annotate('',
            xy=(x_end, BY),
            xytext=(x_start, BY),
            arrowprops=dict(arrowstyle='->',
                            color=C_PHD, lw=1.5,
                            connectionstyle='arc3,rad=-0.32'))
ax.text((x_start+x_end)/2, BY - 0.72,
        'Tracker feedback (predicted DOAs)',
        ha='center', va='center', fontsize=14,
        color=C_PHD, style='italic')

def note(x, y, txt, fc, ec, fs=12):
    ax.text(x, y, txt, ha='center', va='center',
            fontsize=fs, color=C_TEXT, zorder=5,
            bbox=dict(facecolor=fc, edgecolor=ec,
                      boxstyle='round,pad=0.32', alpha=0.93))

note(W*0.27, H - 0.32,
     r'Inference:  SSM.step$(\mathbf{o}_t,\mathbf{h}_{t-1})$'
     r'  $\rightarrow$  $O(d_s)$/scan  $\rightarrow$  3.4 ms on STM32H7',
     '#EAF4EA', '#7DCEA0', fs=12)

note(W*0.74, H - 0.32,
     r'Training:  SSM.forward$(\mathbf{o}_{1:T})$'
     r'  $\rightarrow$  BPTT  $\rightarrow$  42 K params / 41.4 KB INT8',
     '#FEF5E7', '#F0B27A', fs=12)

legend_items = [
    mpatches.Patch(color=C_ENV, label='Signal / COP'),
    mpatches.Patch(color=C_SSM, label='SSM Encoder'),
    mpatches.Patch(color=C_AC,  label='Actor-Critic (PPO)'),
    mpatches.Patch(color=C_PHD, label='GM-PHD Filter'),
]
ax.legend(handles=legend_items, loc='lower right',
          fontsize=14, framealpha=0.92, edgecolor='#CCCCCC',
          bbox_to_anchor=(0.995, 0.02))

fig.savefig('fig_architecture.pdf', dpi=200,
            bbox_inches='tight', facecolor='white')
fig.savefig('fig_architecture.png', dpi=200,
            bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: fig_architecture.pdf / .png")
