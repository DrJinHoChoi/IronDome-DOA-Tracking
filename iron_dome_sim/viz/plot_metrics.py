"""Performance metrics visualization."""

import numpy as np
import matplotlib.pyplot as plt


def plot_rmse_vs_snr(results, title=None, save_path=None):
    """Plot RMSE vs SNR for multiple algorithms with CRLB.

    Args:
        results: Dict from MonteCarloRunner.evaluate_doa().
        title: Plot title.
        save_path: File path to save.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h']
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    idx = 0
    for name, data in results.items():
        if name == 'CRLB':
            continue
        snr = data['snr']
        rmse = np.degrees(data['rmse'])  # Convert to degrees
        ax.semilogy(snr, rmse, marker=markers[idx % len(markers)],
                     color=colors[idx], linewidth=2, markersize=6,
                     label=name)
        idx += 1

    # Plot CRLB
    if 'CRLB' in results:
        crlb_data = results['CRLB']
        crlb_deg = np.degrees(crlb_data['rmse'])
        ax.semilogy(crlb_data['snr'], crlb_deg, 'k--', linewidth=2,
                     label='CRLB')

    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('RMSE (degrees)', fontsize=12)
    ax.set_title(title or 'DOA Estimation RMSE vs SNR', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_gospa(results, dt=1.0, title=None, save_path=None):
    """Plot GOSPA metric over time for multiple trackers.

    Args:
        results: Dict from MonteCarloRunner.evaluate_tracking().
        dt: Time step for x-axis.
        title: Plot title.
        save_path: File path to save.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for idx, (name, data) in enumerate(results.items()):
        gospa = data['gospa']
        if len(gospa) > 0:
            t = np.arange(len(gospa)) * dt
            ax.plot(t, gospa, color=colors[idx], linewidth=2, label=name)

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('GOSPA', fontsize=12)
    ax.set_title(title or 'Multi-Target Tracking GOSPA', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_comparison_bar(metrics_dict, metric_name="RMSE",
                        title=None, save_path=None):
    """Bar chart comparing algorithm performance.

    Args:
        metrics_dict: Dict {algorithm_name: metric_value}.
        metric_name: Name of the metric.
        title: Plot title.
        save_path: File path to save.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    names = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))

    # Highlight the best (lowest) value
    best_idx = np.argmin(values)
    bar_colors = list(colors)
    bar_colors[best_idx] = plt.cm.Set1(0.0)  # Red for best

    bars = ax.bar(names, values, color=bar_colors, edgecolor='black',
                  linewidth=0.5)

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)

    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(title or f'{metric_name} Comparison', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
