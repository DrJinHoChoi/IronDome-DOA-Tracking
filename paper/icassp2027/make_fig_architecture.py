"""Architecture figure — Mamba-COP-RL.

Redesigned for IEEE SPL figure* (full page width).
Three-tier layout:
  TOP:    data-flow pipeline (boxes + arrows)
  MIDDLE: SSM detail (inference step vs. training BPTT)
  BOTTOM: dimension / cost annotations
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── Canvas ────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 4.2))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 14)
ax.set_ylim(0, 4.2)
ax.axis('off')

# ── Palette (muted, IEEE-friendly) ───────────────────────
C_ENV  = '#1A4F8A'   # dark navy — input / COP
C_SSM  = '#1E6845'   # dark green — SSM encoder
C_AC   = '#7B4F1A'   # dark brown — actor-critic
C_PHD  = '#5C2A6E'   # dark purple — GM-PHD
C_BG   = '#F7F7F7'
C_EDGE = '#CCCCCC'
C_TEXT = '#1A1A1A'
C_DIM  = '#555555'   # dimension annotation
C_INF  = '#1E6845'   # inference note
C_TRN  = '#7B3F00'   # training note

# ── Helper: rounded box ───────────────────────────────────
def box(x, y, w, h, title, sub='', color='#555555', tfs=10, sfs=8):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle='round,pad=0.1',
                          facecolor=color, edgecolor='white',
                          linewidth=1.8, alpha=0.88, zorder=3)
    ax.add_patch(rect)
    ty = y + h/2 + (0.13 if sub else 0)
    ax.text(x + w/2, ty, title, ha='center', va='center',
            fontsize=tfs, fontweight='bold', color='white', zorder=4)
    if sub:
        ax.text(x + w/2, y + h/2 - 0.2, sub, ha='center', va='center',
                fontsize=sfs, color='#D0D0D0', zorder=4)

# ── Helper: horizontal arrow with label ──────────────────
def harrow(x0, x1, y, label='', lw=1.8, color='#333333', lfs=8):
    ax.annotate('', xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, mutation_scale=14))
    if label:
        ax.text((x0+x1)/2, y + 0.17, label,
                ha='center', va='bottom', fontsize=lfs,
                color=C_DIM,
                bbox=dict(facecolor='white', edgecolor='none',
                          alpha=0.85, pad=1.5))

# ── Helper: note box ─────────────────────────────────────
def note(x, y, text, fc, ec, fs=7.8):
    ax.text(x, y, text, ha='center', va='center', fontsize=fs,
            color=C_TEXT, zorder=5,
            bbox=dict(facecolor=fc, edgecolor=ec,
                      boxstyle='round,pad=0.35', alpha=0.95))

# ═════════════════════════════════════════════════════════
# TOP ROW — Pipeline blocks  (y=2.2 .. 3.4)
# ═════════════════════════════════════════════════════════
BH = 1.15   # box height
BY = 2.05   # box bottom y

# Block positions (x_left, width)
POS = [
    (0.30, 1.60),   # 0: Sensor Array
    (2.50, 1.70),   # 1: COP Spectrum
    (4.90, 2.20),   # 2: SSM Encoder  (wider)
    (7.80, 1.90),   # 3: Actor-Critic
    (10.40, 1.90),  # 4: GM-PHD
]

box(*POS[0], BY, BH, 'Sensor Array',
    r'$M{=}8$ ULA', C_ENV, tfs=10)
box(*POS[1], BY, BH, 'COP Spectrum',
    r'$N_\theta{=}181$', C_ENV, tfs=10)
box(*POS[2], BY, BH, 'SSM Encoder',
    r'$d_h{=}48,\;d_s{=}24$', C_SSM, tfs=10)
box(*POS[3], BY, BH, 'Actor-Critic',
    r'MLP $183{\to}64{\to}64{\to}3$', C_AC, tfs=10)
box(*POS[4], BY, BH, 'GM-PHD Filter',
    r'$w_b,\tau_p,p_D$', C_PHD, tfs=10)

# Arrows between blocks (right edge → left edge of next)
cy = BY + BH/2
harrow(POS[0][0]+POS[0][1], POS[1][0], cy,
       r'$\mathbf{X}\!\in\!\mathbb{C}^{M\times L}$')
harrow(POS[1][0]+POS[1][1], POS[2][0], cy,
       r'$\mathbf{o}_t\!\in\!\mathbb{R}^{183}$')
harrow(POS[2][0]+POS[2][1], POS[3][0], cy,
       r'$\tilde{\mathbf{o}}_t\!\in\!\mathbb{R}^{183}$')
harrow(POS[3][0]+POS[3][1], POS[4][0], cy,
       r'$\mathbf{a}_t\!\in\![-1,1]^3$')

# GM-PHD output label
ax.text(POS[4][0]+POS[4][1]+0.15, cy,
        r'$\hat{\Theta}_t$', ha='left', va='center',
        fontsize=9, color=C_PHD, fontweight='bold')
ax.annotate('', xy=(POS[4][0]+POS[4][1]+0.12, cy),
            xytext=(POS[4][0]+POS[4][1], cy),
            arrowprops=dict(arrowstyle='->', color=C_PHD, lw=1.5))

# ── Hidden state self-loop (below SSM encoder) ────────────
sx = POS[2][0]; sw = POS[2][1]
ax.annotate('',
            xy=(sx, BY + 0.32),
            xytext=(sx + sw, BY + 0.32),
            arrowprops=dict(arrowstyle='<->',
                            color=C_SSM, lw=1.6,
                            connectionstyle='arc3,rad=0.50'))
ax.text(sx + sw/2, BY - 0.35,
        r'$\mathbf{h}_t \in \mathbb{R}^{d_s}$  (selective hidden state)',
        ha='center', va='center', fontsize=8.5,
        color=C_SSM, style='italic')

# ── Feedback: GM-PHD → COP (curved under) ────────────────
ax.annotate('',
            xy=(POS[1][0]+POS[1][1]/2, BY),
            xytext=(POS[4][0]+POS[4][1]/2, BY),
            arrowprops=dict(arrowstyle='->',
                            color=C_PHD, lw=1.4,
                            connectionstyle='arc3,rad=-0.38'))
ax.text(7.0, 1.08,
        'Tracker feedback (predicted DOAs)',
        ha='center', va='center', fontsize=8,
        color=C_PHD, style='italic')

# ═════════════════════════════════════════════════════════
# BOTTOM ROW — Inference vs Training notes
# ═════════════════════════════════════════════════════════
note(3.55, 0.47,
     r'Inference:  SSM.step$(\mathbf{o}_t,\mathbf{h}_{t-1})$  '
     r'$\rightarrow$  $O(d_s)$ per scan  $\rightarrow$  3.4 ms / scan on STM32H7',
     '#EEF4EE', '#99CC99', fs=8.5)

note(10.10, 0.47,
     r'Training:  SSM.forward$(\mathbf{o}_{1:T})$  '
     r'$\rightarrow$  BPTT through encoder  $\rightarrow$  42 K params / 41.4 KB INT8',
     '#FDF5EE', '#DDAA77', fs=8.5)

# ── Section label ─────────────────────────────────────────
ax.text(7.0, 3.95, 'Mamba-COP-RL Pipeline',
        ha='center', va='center', fontsize=12,
        fontweight='bold', color=C_TEXT)

# ── Legend ────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(color=C_ENV, label='Signal / COP'),
    mpatches.Patch(color=C_SSM, label='SSM Encoder (Mamba)'),
    mpatches.Patch(color=C_AC,  label='Actor-Critic (PPO)'),
    mpatches.Patch(color=C_PHD, label='GM-PHD Filter'),
]
ax.legend(handles=legend_items, loc='upper right',
          fontsize=8.5, framealpha=0.9, edgecolor=C_EDGE,
          bbox_to_anchor=(0.995, 0.98))

for ext in ('pdf', 'png'):
    fig.savefig(f'fig_architecture.{ext}', dpi=200,
                bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: fig_architecture.pdf / .png")
