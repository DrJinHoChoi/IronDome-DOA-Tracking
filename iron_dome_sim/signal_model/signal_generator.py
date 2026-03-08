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
        signal_type: Signal characteristics string. Options:
                     "stationary" - complex Gaussian (i.i.d.)
                     "non_stationary" - amplitude-modulated radar returns
                     "missile" - Doppler-shifted with RCS fluctuation
                     "fm" - frequency modulation (wideband-like)
                     "chirp" - linear frequency modulation (LFM radar)
                     "psk" - phase shift keying (communication signal)
                     "speech" - speech-like quasi-periodic signal
                     "mixed" - each source randomly chosen from above types
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

    elif signal_type == "fm":
        # Frequency Modulation: wideband-like signal with time-varying
        # instantaneous frequency. Deterministic/structured => non-Gaussian.
        # S[k](t) = exp(j*(2*pi*f_c*t + beta*sin(2*pi*f_m*t)))
        S = np.zeros((K, T), dtype=complex)
        for k in range(K):
            t = np.arange(T)
            f_c = np.random.uniform(0.1, 0.4)       # carrier frequency
            f_m = np.random.uniform(0.005, 0.03)     # modulation frequency
            beta = np.random.uniform(1.0, 5.0)       # modulation index
            phase0 = np.random.uniform(0, 2 * np.pi)
            S[k] = np.exp(1j * (2 * np.pi * f_c * t
                                + beta * np.sin(2 * np.pi * f_m * t)
                                + phase0))

    elif signal_type == "chirp":
        # Linear Frequency Modulation (LFM): most common radar waveform.
        # Deterministic quadratic phase => non-Gaussian.
        # S[k](t) = (1 + 0.3*sin(2*pi*f_am*t)) * exp(j*pi*mu*t^2)
        S = np.zeros((K, T), dtype=complex)
        for k in range(K):
            t = np.arange(T)
            mu = np.random.uniform(0.01, 0.1)        # chirp rate
            f_am = np.random.uniform(0.001, 0.01)    # AM envelope frequency
            phase0 = np.random.uniform(0, 2 * np.pi)
            # Slow AM envelope for amplitude variation
            envelope = 1.0 + 0.3 * np.sin(2 * np.pi * f_am * t)
            S[k] = envelope * np.exp(1j * (np.pi * mu * t ** 2 + phase0))

    elif signal_type == "psk":
        # Phase Shift Keying (BPSK/QPSK communication signal).
        # Discrete-alphabet phase symbols => non-Gaussian.
        # Random symbols from {0, pi/2, pi, 3*pi/2} with slow power variation.
        S = np.zeros((K, T), dtype=complex)
        for k in range(K):
            t = np.arange(T)
            symbol_rate = np.random.uniform(0.01, 0.05)
            # Number of samples per symbol
            samples_per_symbol = max(1, int(1.0 / symbol_rate))
            n_symbols = int(np.ceil(T / samples_per_symbol))
            # Random QPSK phase symbols
            phase_alphabet = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2])
            symbol_phases = np.random.choice(phase_alphabet, size=n_symbols)
            # Repeat each symbol for its duration, then truncate to T
            phase_seq = np.repeat(symbol_phases, samples_per_symbol)[:T]
            # Slow sinusoidal power variation
            f_pow = np.random.uniform(0.001, 0.01)
            power_env = 1.0 + 0.5 * np.sin(2 * np.pi * f_pow * t)
            S[k] = np.sqrt(np.abs(power_env)) * np.exp(1j * phase_seq)

    elif signal_type == "speech":
        # Speech-like signal: quasi-periodic voiced segments + noise bursts.
        # Models key properties of real speech:
        #   1. Quasi-periodic (pitch period varies slowly)
        #   2. Formant structure (resonant frequencies)
        #   3. Time-varying amplitude (syllable envelope)
        #   4. Voiced/unvoiced alternation
        # Non-Gaussian: structured, deterministic-like during voiced segments.
        S = np.zeros((K, T), dtype=complex)
        for k in range(K):
            t = np.arange(T)
            # Fundamental frequency (pitch): 80-300 Hz normalized
            f0 = np.random.uniform(0.02, 0.08)
            # Slow pitch variation (vibrato/prosody)
            pitch_mod = np.random.uniform(0.001, 0.005)
            pitch_depth = np.random.uniform(0.1, 0.3)
            inst_f0 = f0 * (1.0 + pitch_depth * np.sin(2 * np.pi * pitch_mod * t))

            # Glottal pulse train (voiced excitation)
            phase_acc = np.cumsum(2 * np.pi * inst_f0)
            glottal = np.sin(phase_acc) + 0.5 * np.sin(2 * phase_acc)

            # Formant filtering: 2-3 resonances (simplified)
            f1 = np.random.uniform(0.05, 0.15)  # 1st formant
            f2 = np.random.uniform(0.15, 0.30)  # 2nd formant
            formant1 = np.sin(2 * np.pi * f1 * t)
            formant2 = 0.5 * np.sin(2 * np.pi * f2 * t)

            # Syllable envelope (amplitude modulation ~3-7 Hz = slow)
            syl_rate = np.random.uniform(0.003, 0.01)
            envelope = np.abs(np.sin(2 * np.pi * syl_rate * t)) ** 0.5
            envelope = np.maximum(envelope, 0.1)  # minimum level

            # Voiced/unvoiced mixing
            vu_rate = np.random.uniform(0.002, 0.008)
            vu_mask = (np.sin(2 * np.pi * vu_rate * t) > -0.3).astype(float)
            unvoiced = (np.random.randn(T) + 1j * np.random.randn(T)) * 0.3

            # Combine: voiced (glottal + formants) + unvoiced noise
            voiced = glottal * (1.0 + formant1 + formant2)
            signal = envelope * (vu_mask * voiced + (1 - vu_mask) * np.abs(unvoiced))

            # Make complex analytic signal
            phase0 = np.random.uniform(0, 2 * np.pi)
            S[k] = signal * np.exp(1j * (phase_acc * 0.5 + phase0))

            # Normalize
            S[k] = S[k] / (np.std(S[k]) + 1e-10)

    elif signal_type == "mixed":
        # Each source gets a randomly chosen signal type.
        # Delegates per-source generation to this same function.
        available_types = ["fm", "chirp", "psk", "missile", "non_stationary",
                           "speech"]
        S = np.zeros((K, T), dtype=complex)
        for k in range(K):
            chosen = np.random.choice(available_types)
            S_single = _generate_source_signals(1, T, chosen)
            S[k] = S_single[0]

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
