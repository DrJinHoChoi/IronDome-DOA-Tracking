"""Spatial spectrum utility functions."""

import numpy as np


def find_peaks_doa(spectrum, scan_angles, num_peaks):
    """Find DOA estimates from spatial spectrum peaks.

    Args:
        spectrum: Spatial spectrum values.
        scan_angles: Corresponding angles in radians.
        num_peaks: Number of peaks to find.

    Returns:
        doa_estimates: Estimated DOA angles in radians, sorted.
    """
    # Find local maxima
    peaks = []
    for i in range(1, len(spectrum) - 1):
        if spectrum[i] > spectrum[i - 1] and spectrum[i] > spectrum[i + 1]:
            peaks.append((spectrum[i], scan_angles[i]))

    # Sort by peak height (descending)
    peaks.sort(key=lambda x: x[0], reverse=True)

    # Take top num_peaks, with minimum separation
    min_sep = np.radians(1.0)  # minimum 1 degree separation
    selected = []
    for height, angle in peaks:
        if len(selected) >= num_peaks:
            break
        # Check separation from already selected peaks
        if all(abs(angle - s) > min_sep for s in selected):
            selected.append(angle)

    return np.sort(np.array(selected))
