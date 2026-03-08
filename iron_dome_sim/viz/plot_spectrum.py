"""DOA spatial spectrum visualization."""

import numpy as np
import matplotlib.pyplot as plt


def plot_doa_spectrum(spectrum, scan_angles, true_doas=None,
                      estimated_doas=None, title=None, save_path=None):
    """Plot a single DOA spatial spectrum.

    Args:
        spectrum: Spatial spectrum values.
        scan_angles: Scan angles in radians.
        true_doas: True DOA angles in radians (for reference lines).
        estimated_doas: Estimated DOA angles (for markers).
        title: Plot title.
        save_path: File path to save.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    scan_deg = np.degrees(scan_angles)
    ax.plot(scan_deg, 10 * np.log10(spectrum + 1e-15), 'b-', linewidth=1.5)

    if true_doas is not None:
        for doa in true_doas:
            ax.axvline(np.degrees(doa), color='r', linestyle='--',
                       alpha=0.7, linewidth=1)

    if estimated_doas is not None:
        for doa in estimated_doas:
            ax.axvline(np.degrees(doa), color='g', linestyle=':',
                       alpha=0.8, linewidth=1.5)

    ax.set_xlabel('Angle (degrees)')
    ax.set_ylabel('Spectrum (dB)')
    ax.set_title(title or 'DOA Spatial Spectrum')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-90, 90])

    from matplotlib.lines import Line2D
    legend = [Line2D([0], [0], color='b', label='Spectrum')]
    if true_doas is not None:
        legend.append(Line2D([0], [0], color='r', linestyle='--', label='True DOA'))
    if estimated_doas is not None:
        legend.append(Line2D([0], [0], color='g', linestyle=':', label='Estimated DOA'))
    ax.legend(handles=legend)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_spectrum_comparison(spectra_dict, scan_angles, true_doas,
                             title=None, save_path=None):
    """Compare spatial spectra from multiple DOA algorithms.

    Args:
        spectra_dict: Dict {algorithm_name: spectrum_values}.
        scan_angles: Scan angles in radians.
        true_doas: True DOA angles in radians.
        title: Plot title.
        save_path: File path to save.
    """
    n_algs = len(spectra_dict)
    fig, axes = plt.subplots(n_algs, 1, figsize=(12, 3 * n_algs), sharex=True)

    if n_algs == 1:
        axes = [axes]

    scan_deg = np.degrees(scan_angles)
    colors = plt.cm.tab10(np.linspace(0, 1, n_algs))

    for idx, (name, spectrum) in enumerate(spectra_dict.items()):
        ax = axes[idx]

        if spectrum is not None:
            spec_db = 10 * np.log10(spectrum + 1e-15)
            ax.plot(scan_deg, spec_db, color=colors[idx], linewidth=1.5)

            for doa in true_doas:
                ax.axvline(np.degrees(doa), color='r', linestyle='--',
                           alpha=0.5, linewidth=1)

            ax.set_ylabel('dB')
        else:
            ax.text(0.5, 0.5, 'No spectrum (direct estimation)',
                    transform=ax.transAxes, ha='center', va='center')

        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-90, 90])

    axes[-1].set_xlabel('Angle (degrees)')
    plt.suptitle(title or 'DOA Spectrum Comparison', fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
