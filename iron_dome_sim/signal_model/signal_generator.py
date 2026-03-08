"""Signal generation for radar array processing.

Generates received signal snapshots with configurable source signals,
noise levels, and non-stationarity characteristics.
"""

import numpy as np


def generate_snapshots(array, thetas, snr_db, T, signal_type="non_stationary",
                       phis=None):
    """Generate received signal snapshots at the array.

    Model: x(t) = A(theta) * s(t) + n(t)

    Args:
        array: Array object (ULA or URA) with steering_matrix method.
        thetas: Source DOA azimuth angles in radians, shape (K,).
        snr_db: Signal-to-noise ratio in dB (per source).
        T: Number of time snapshots.
        signal_type: "stationary" (complex Gaussian) or "non_stationary"
                     (amplitude-modulated, realistic for radar returns).
        phis: Source elevation angles in radians (for URA), shape (K,).

    Returns:
        X: Received signal matrix, shape (M, T).
        S: Source signal matrix, shape (K, T).
        N: Noise matrix, shape (M, T).
    """
    K = len(thetas)
    M = array.M

    # Steering matrix
    if phis is not None:
        A = array.steering_matrix(thetas, phis)
    else:
        A = array.steering_matrix(thetas)

    # Source signals
    S = _generate_source_signals(K, T, signal_type)

    # Scale signals to achieve desired SNR
    signal_power = np.mean(np.abs(S) ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))

    # Noise: complex Gaussian
    N = np.sqrt(noise_power / 2) * (
        np.random.randn(M, T) + 1j * np.random.randn(M, T)
    )

    # Received signal
    X = A @ S + N

    return X, S, N


def _generate_source_signals(K, T, signal_type):
    """Generate source signal matrix.

    Args:
        K: Number of sources.
        T: Number of snapshots.
        signal_type: Signal characteristics.

    Returns:
        S: Source signal matrix, shape (K, T).
    """
    if signal_type == "stationary":
        # Complex Gaussian (i.i.d.)
        S = (np.random.randn(K, T) + 1j * np.random.randn(K, T)) / np.sqrt(2)

    elif signal_type == "non_stationary":
        # Amplitude-modulated signals (non-Gaussian, non-stationary)
        # Models realistic radar returns from maneuvering targets
        S = np.zeros((K, T), dtype=complex)
        for k in range(K):
            # Random carrier frequency for each source
            freq = np.random.uniform(0.05, 0.4)
            phase = np.random.uniform(0, 2 * np.pi)
            t = np.arange(T)

            # Amplitude modulation (slow variation)
            am_freq = np.random.uniform(0.001, 0.02)
            amplitude = 1.0 + 0.5 * np.sin(2 * np.pi * am_freq * t)

            # Phase modulation
            S[k] = amplitude * np.exp(1j * (2 * np.pi * freq * t + phase))

    elif signal_type == "missile":
        # Missile radar return: Doppler-shifted with RCS fluctuation
        S = np.zeros((K, T), dtype=complex)
        for k in range(K):
            t = np.arange(T)
            # Doppler frequency (higher for fast targets)
            doppler = np.random.uniform(0.1, 0.45)
            # Swerling I RCS fluctuation
            rcs = np.random.exponential(1.0, T)
            phase = np.random.uniform(0, 2 * np.pi)
            S[k] = np.sqrt(rcs) * np.exp(1j * (2 * np.pi * doppler * t + phase))

    else:
        raise ValueError(f"Unknown signal_type: {signal_type}")

    return S


def generate_multi_radar_snapshots(radar_network, target_positions, snr_db, T,
                                   signal_type="missile"):
    """Generate snapshots for multiple radars observing multiple targets.

    Args:
        radar_network: List of (array, position) tuples.
        target_positions: Target 3D positions, shape (K, 3).
        snr_db: SNR in dB.
        T: Number of snapshots per scan.
        signal_type: Signal type string.

    Returns:
        List of (X, thetas, phis) tuples, one per radar.
    """
    results = []

    for array, radar_pos in radar_network:
        # Compute DOA angles from radar to each target
        rel_pos = target_positions - radar_pos  # (K, 3)
        ranges = np.linalg.norm(rel_pos, axis=1)

        # Azimuth: atan2(y, x)
        thetas = np.arctan2(rel_pos[:, 1], rel_pos[:, 0])
        # Elevation: asin(z / range)
        phis = np.arcsin(rel_pos[:, 2] / (ranges + 1e-10))

        # Range-dependent SNR adjustment
        ref_range = 10000.0  # reference range in meters
        snr_adjusted = snr_db - 40 * np.log10(ranges / ref_range)
        avg_snr = np.mean(snr_adjusted)

        # Generate snapshots
        if hasattr(array, 'My'):  # URA
            X, S, N = generate_snapshots(array, thetas, avg_snr, T,
                                         signal_type, phis)
        else:  # ULA
            X, S, N = generate_snapshots(array, thetas, avg_snr, T,
                                         signal_type)

        results.append({
            'X': X,
            'thetas': thetas,
            'phis': phis,
            'ranges': ranges,
            'snr_adjusted': snr_adjusted,
        })

    return results
