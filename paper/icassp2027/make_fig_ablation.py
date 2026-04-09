"""Ablation figure — wide version for IEEE SPL figure*."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(13, 4.0))
plt.subplots_adjust(wspace=0.32)

C_FIXED = '#8B2252'
C_MLP   = '#1A4F8A'
C_MAMBA = '#1E6845'
C_TEXT  = '#1A1A1A'

methods  = ['Fixed', 'MLP-PPO', 'Mamba-COP-RL']
colors   = [C_FIXED, C_MLP, C_MAMBA]
gospa_12  = [0.2898, 0.2680, 0.3150]
gospa_183 = [0.2599, 0.2241, 0.2130]

for idx, (ax, gospa, title, ylim) in enumerate(zip(
    axes,
    [gospa_12, gospa_183],
    ['12-dim hand-crafted statistics  ($T = 40$ scans)',
     '183-dim raw COP spectrum  ($T = 200$ scans)'],
    [(0.22, 0.37), (0.16, 0.30)]
)):
    ax.set_facecolor('#FFFFFF')
    ax.set_title(title, fontsize=11, fontweight='bold',
                 color=C_TEXT, pad=8)

    bars = ax.bar(methods, gospa, color=colors, width=0.52,
                  edgecolor='white', linewidth=1.5, alpha=0.85)

    for bar, v in zip(bars, gospa):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.004,
                f'{v:.4f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold', color=C_TEXT)

    if idx == 0:
        ax.annotate('MLP wins  $-7.5\%$',
                    xy=(1, gospa[1]), xytext=(1, gospa[1] - 0.045),
                    ha='center', fontsize=9.5, color=C_MLP,
                    fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=C_MLP, lw=1.4))
        ax.annotate('Mamba  $+8.7\%$ worse',
                    xy=(2, gospa[2]), xytext=(2, gospa[2] + 0.020),
                    ha='center', fontsize=9, color='#5A3030',
                    arrowprops=dict(arrowstyle='->', color='#5A3030', lw=1.2))
        ax.text(0.5, 0.97,
                r'$f(\mathbf{o}_t) \approx$ sufficient statistic  '
                r'$\Rightarrow$  MLP sufficient',
                transform=ax.transAxes, ha='center', va='top',
                fontsize=9, color='#3D2B1F', style='italic',
                bbox=dict(facecolor='#F5F0E8', edgecolor='#C8B89A',
                          boxstyle='round,pad=0.35', alpha=0.9))
    else:
        ax.annotate('Mamba wins\n$-18.0\%$ vs Fixed\n$-5.0\%$ vs MLP',
                    xy=(2, gospa[2]), xytext=(2, gospa[2] - 0.038),
                    ha='center', fontsize=9.5, color=C_MAMBA,
                    fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=C_MAMBA, lw=1.4))
        ax.text(0.5, 0.97,
                r'$\mathcal{I}(\mathbf{o}_t;\mathbf{o}_{1:t-1}) \gg '
                r'\mathcal{I}(f(\mathbf{o}_t);\mathbf{o}_{1:t-1})$  '
                r'$\Rightarrow$  SSM encoder helps',
                transform=ax.transAxes, ha='center', va='top',
                fontsize=9, color='#1A2A4A', style='italic',
                bbox=dict(facecolor='#EDF0F7', edgecolor='#8899BB',
                          boxstyle='round,pad=0.35', alpha=0.9))

    ax.set_ylabel('Mean GOSPA  ($\\downarrow$ better)', fontsize=10,
                  color=C_TEXT)
    ax.set_ylim(*ylim)
    ax.tick_params(labelsize=9.5, colors=C_TEXT)
    ax.grid(axis='y', alpha=0.22, color='#AAAAAA')
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_color('#444444')

legend_items = [
    mpatches.Patch(color=C_FIXED,  label='Fixed threshold'),
    mpatches.Patch(color=C_MLP,    label='MLP-PPO'),
    mpatches.Patch(color=C_MAMBA,  label='Mamba-COP-RL'),
]
fig.legend(handles=legend_items, loc='lower center', ncol=3,
           fontsize=10, framealpha=0.95, edgecolor='#BBBBBB',
           bbox_to_anchor=(0.5, -0.04))

plt.suptitle('Ablation: Observation Dimensionality Determines SSM Advantage'
             '  (Proposition\u00a01)',
             fontsize=12, fontweight='bold', color=C_TEXT, y=1.02)

for ext in ('pdf', 'png'):
    plt.savefig(f'fig_ablation.{ext}', dpi=200, bbox_inches='tight',
                facecolor='white')
plt.close()
print("Saved: fig_ablation.pdf / .png")
