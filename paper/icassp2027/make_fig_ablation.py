"""Ablation figure — wide version for IEEE SPL figure*."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(13, 3.1))
plt.subplots_adjust(wspace=0.32, bottom=0.10)

C_FIXED = '#C0392B'
C_MLP   = '#2980B9'
C_MAMBA = '#27AE60'
C_TEXT  = '#1A1A1A'

methods  = ['Fixed', 'MLP-PPO', 'Mamba-COP-RL']
colors   = [C_FIXED, C_MLP, C_MAMBA]
gospa_12  = [0.2898, 0.2680, 0.3150]
gospa_183 = [0.2599, 0.2241, 0.2130]

for idx, (ax, gospa, title, ylim) in enumerate(zip(
    axes,
    [gospa_12, gospa_183],
    ['(a)  12-dim hand-crafted statistics  ($T = 40$ scans)',
     '(b)  183-dim raw COP spectrum  ($T = 200$ scans)'],
    [(0.22, 0.37), (0.16, 0.30)]
)):
    ax.set_facecolor('#FFFFFF')
    ax.set_title(title, fontsize=13, fontweight='bold',
                 color=C_TEXT, pad=8)

    bars = ax.bar(methods, gospa, color=colors, width=0.52,
                  edgecolor='white', linewidth=1.5, alpha=0.85)

    for bar, v in zip(bars, gospa):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.004,
                f'{v:.4f}', ha='center', va='bottom',
                fontsize=12, fontweight='bold', color=C_TEXT)

    if idx == 0:
        ax.annotate('MLP wins  $-7.5\%$',
                    xy=(1, gospa[1]), xytext=(1, gospa[1] - 0.045),
                    ha='center', fontsize=11.5, color=C_TEXT,
                    fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=C_MLP, lw=1.4))
        ax.annotate('Mamba  $+8.7\%$ worse',
                    xy=(2, gospa[2]), xytext=(2, gospa[2] + 0.020),
                    ha='center', fontsize=11, color=C_TEXT,
                    arrowprops=dict(arrowstyle='->', color=C_FIXED, lw=1.2))
        ax.text(0.5, 0.97,
                r'$f(\mathbf{o}_t) \approx$ sufficient statistic  '
                r'$\Rightarrow$  MLP sufficient',
                transform=ax.transAxes, ha='center', va='top',
                fontsize=11, color=C_TEXT, style='italic',
                bbox=dict(facecolor='#F5F0E8', edgecolor='#C8B89A',
                          boxstyle='round,pad=0.35', alpha=0.9))
    else:
        ax.annotate('Mamba wins\n$-18.0\%$ vs Fixed\n$-5.0\%$ vs MLP',
                    xy=(2, gospa[2]), xytext=(2, gospa[2] - 0.038),
                    ha='center', fontsize=11.5, color=C_TEXT,
                    fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=C_MAMBA, lw=1.4))
        ax.text(0.5, 0.97,
                r'$\mathcal{I}(\mathbf{o}_t;\mathbf{o}_{1:t-1}) \gg '
                r'\mathcal{I}(f(\mathbf{o}_t);\mathbf{o}_{1:t-1})$  '
                r'$\Rightarrow$  SSM encoder helps',
                transform=ax.transAxes, ha='center', va='top',
                fontsize=11, color=C_TEXT, style='italic',
                bbox=dict(facecolor='#EDF0F7', edgecolor='#8899BB',
                          boxstyle='round,pad=0.35', alpha=0.9))

    ax.set_ylabel('Mean GOSPA  ($\\downarrow$ better)', fontsize=12,
                  color=C_TEXT)
    ax.set_ylim(*ylim)
    ax.tick_params(labelsize=11.5, colors=C_TEXT)
    ax.grid(axis='y', alpha=0.22, color='#AAAAAA')
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_color('#444444')

legend_items = [
    mpatches.Patch(color=C_FIXED,  label='Fixed'),
    mpatches.Patch(color=C_MLP,    label='MLP-PPO'),
    mpatches.Patch(color=C_MAMBA,  label='Mamba-COP-RL'),
]
axes[1].legend(handles=legend_items, loc='upper left',
               fontsize=11.5, framealpha=0.95, edgecolor='#BBBBBB')

for ext in ('pdf', 'png'):
    plt.savefig(f'fig_ablation.{ext}', dpi=200, bbox_inches='tight',
                facecolor='white')
plt.close()
print("Saved: fig_ablation.pdf / .png")
