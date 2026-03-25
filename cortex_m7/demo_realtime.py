"""COP-RFS Real-Time Demo: Cortex-M7 Simulation.

Simulates the full COP-RFS pipeline processing radar scans
in real-time, demonstrating:
  1. Underdetermined DOA estimation (K > M-1)
  2. T-COP temporal cumulant accumulation
  3. Physics-based GM-PHD tracking through target crossings
  4. COP-spectrum birth for new target detection
  5. Tracker → COP feedback loop convergence
  6. Cortex-M7 latency profiling

Usage:
    python demo_realtime.py
"""

import numpy as np
import time
from cop_rfs_rt import COP_RFS_RT, print_memory_map


def generate_ula_signal(M, T, doas_deg, snr_db=15, signal_type='bpsk'):
    """Generate ULA received signal with K narrowband sources.

    Args:
        M: Number of sensors
        T: Number of snapshots
        doas_deg: True DOA angles in degrees
        snr_db: Signal-to-noise ratio
        signal_type: 'bpsk' (non-Gaussian, required for COP)

    Returns:
        X: Received signal (M, T) complex
    """
    K = len(doas_deg)
    doas_rad = np.radians(doas_deg)

    # Steering matrix A (M × K)
    A = np.zeros((M, K), dtype=np.complex64)
    for k in range(K):
        for m in range(M):
            A[m, k] = np.exp(1j * np.pi * m * np.sin(doas_rad[k]))

    # Source signals (BPSK: non-Gaussian, maximizes COP advantage)
    if signal_type == 'bpsk':
        S = (2 * (np.random.randint(0, 2, (K, T))) - 1).astype(np.complex64)
    elif signal_type == 'qpsk':
        phase = np.random.choice([0, np.pi/2, np.pi, 3*np.pi/2], (K, T))
        S = np.exp(1j * phase).astype(np.complex64)
    else:
        S = (np.random.randn(K, T) + 1j * np.random.randn(K, T)).astype(np.complex64)

    # Signal power
    sig_power = np.float32(10 ** (snr_db / 10))
    S *= np.sqrt(sig_power)

    # Received signal: X = A @ S + N
    X = A @ S
    noise = (np.random.randn(M, T) + 1j * np.random.randn(M, T)).astype(np.complex64)
    noise *= np.float32(1.0 / np.sqrt(2))
    X += noise

    return X


def demo_static_underdetermined():
    """Demo 1: Static underdetermined DOA estimation (K=10 > M-1=7)."""
    print("\n" + "=" * 60)
    print("DEMO 1: Static Underdetermined DOA (K=10, M=8)")
    print("=" * 60)

    M = 8
    T = 128
    K = 10
    true_doas = np.array([-60, -45, -30, -15, -5, 5, 15, 30, 45, 60], dtype=np.float32)

    pipeline = COP_RFS_RT(M=M, T=T, dt=0.1, n_angles=361, alpha=0.0)

    X = generate_ula_signal(M, T, true_doas, snr_db=20, signal_type='bpsk')

    t0 = time.perf_counter()
    targets, doas_deg, spectrum = pipeline.process_scan(X, K)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    print(f"\nTrue DOAs:      {true_doas}")
    print(f"COP estimates:  {np.round(doas_deg, 1)}")
    print(f"Detected:       {len(doas_deg)}/{K} sources")
    print(f"Latency:        {elapsed_ms:.2f} ms")

    if len(doas_deg) > 0:
        # Match estimated to true DOAs
        errors = []
        for est in doas_deg:
            min_err = min(abs(est - t) for t in true_doas)
            errors.append(min_err)
        rmse = np.sqrt(np.mean(np.array(errors) ** 2))
        print(f"RMSE:           {rmse:.2f}°")


def demo_tracking_crossing():
    """Demo 2: Multi-target tracking through crossing trajectories."""
    print("\n" + "=" * 60)
    print("DEMO 2: Tracking Through Target Crossing (K=3)")
    print("=" * 60)

    M = 8
    T = 64
    K = 3
    dt = 0.1
    n_scans = 50

    pipeline = COP_RFS_RT(M=M, T=T, dt=dt, n_angles=361, alpha=0.85)

    # Trajectories: two crossing, one stationary
    trajectories = []
    for scan in range(n_scans):
        t = scan * dt
        doa1 = -30 + 20 * t    # Moving right
        doa2 = 30 - 20 * t     # Moving left (crosses doa1)
        doa3 = 0.0             # Stationary
        trajectories.append([doa1, doa2, doa3])

    print(f"\nTrajectory: Target 1 (-30° → +70°), Target 2 (30° → -70°), Target 3 (0° fixed)")
    print(f"Crossing occurs at scan ~15 (t=1.5s)")
    print(f"\n{'Scan':>4s} | {'True DOAs':>25s} | {'Est DOAs':>25s} | {'Tracks':>6s} | {'ms':>6s}")
    print("-" * 80)

    latencies = []
    for scan in range(n_scans):
        true_doas = np.array(trajectories[scan], dtype=np.float32)
        true_doas = np.clip(true_doas, -85, 85)  # Stay in FOV
        K_actual = len([d for d in true_doas if -85 <= d <= 85])

        X = generate_ula_signal(M, T, true_doas, snr_db=15, signal_type='bpsk')

        t0 = time.perf_counter()
        targets, doas_deg, spectrum = pipeline.process_scan(X, K_actual)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)

        if scan % 5 == 0 or scan == n_scans - 1:
            true_str = ', '.join(f'{d:6.1f}' for d in true_doas)
            est_str = ', '.join(f'{d:6.1f}' for d in doas_deg[:K_actual])
            n_tracks = len(targets)
            print(f"{scan:4d} | {true_str:>25s} | {est_str:>25s} | {n_tracks:>6d} | {elapsed_ms:6.2f}")

    avg_lat = np.mean(latencies)
    max_lat = np.max(latencies)
    print(f"\nLatency: avg={avg_lat:.2f}ms, max={max_lat:.2f}ms")
    print(f"Cortex-M7 @480MHz estimate: ~{avg_lat * 0.03:.2f}ms avg")  # Python→C speedup ~30x


def demo_capacity_limit():
    """Demo 3: COP at theoretical capacity limit (K=14=rho*(M-1))."""
    print("\n" + "=" * 60)
    print("DEMO 3: Capacity Limit (K=14=rho*(M-1), M=8, rho=2)")
    print("=" * 60)

    M = 8
    T = 256
    K = 14
    true_doas = np.linspace(-65, 65, K).astype(np.float32)

    pipeline = COP_RFS_RT(M=M, T=T, dt=0.1, n_angles=721, alpha=0.0)

    # Run 3 scans for T-COP accumulation
    for scan in range(3):
        X = generate_ula_signal(M, T, true_doas, snr_db=25, signal_type='bpsk')
        t0 = time.perf_counter()
        targets, doas_deg, spectrum = pipeline.process_scan(X, K)
        elapsed_ms = (time.perf_counter() - t0) * 1000

    print(f"\nTrue DOAs ({K}): {np.round(true_doas, 1)}")
    print(f"COP estimates:   {np.round(doas_deg, 1)}")
    print(f"Detected:        {len(doas_deg)}/{K}")
    print(f"Latency:         {elapsed_ms:.2f} ms")


def demo_latency_profiling():
    """Demo 4: Detailed latency breakdown per pipeline stage."""
    print("\n" + "=" * 60)
    print("DEMO 4: Latency Profiling (per stage)")
    print("=" * 60)

    M = 8
    T = 64
    K = 8
    true_doas = np.linspace(-40, 40, K).astype(np.float32)

    pipeline = COP_RFS_RT(M=M, T=T, dt=0.1, n_angles=361, alpha=0.85)
    X = generate_ula_signal(M, T, true_doas, snr_db=15, signal_type='bpsk')

    # Warm-up
    pipeline.process_scan(X, K)

    # Profile each stage
    n_trials = 10
    times = {'cumulant': [], 'eigen': [], 'prior': [],
             'spectrum': [], 'peaks': [], 'predict': [],
             'associate': [], 'update': [], 'prune': []}

    for _ in range(n_trials):
        X = generate_ula_signal(M, T, true_doas, snr_db=15, signal_type='bpsk')

        t0 = time.perf_counter()
        pipeline.cop.compute_cumulant(X)
        times['cumulant'].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        K_actual = pipeline.cop.eigendecompose(K)
        times['eigen'].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        pipeline.cop.apply_tracker_prior(K_actual)
        times['prior'].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        pipeline.cop.compute_spectrum(K_actual)
        times['spectrum'].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        pipeline.cop.find_peaks(K)
        times['peaks'].append(time.perf_counter() - t0)

        n_meas = pipeline.cop.buf.n_peaks
        meas = pipeline.cop.buf.peak_doas[:n_meas]

        t0 = time.perf_counter()
        pipeline.phd.predict()
        times['predict'].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        assoc, unassoc = pipeline.phd.associate(meas, n_meas)
        times['associate'].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        means, covs, weights, labels, n_upd = pipeline.phd.update(
            meas, n_meas, pipeline.cop.buf.spectrum,
            pipeline.cop.buf.scan_angles, assoc, unassoc)
        times['update'].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        pipeline.phd.prune_and_merge(means, covs, weights, labels, n_upd)
        times['prune'].append(time.perf_counter() - t0)

    print(f"\n{'Stage':>15s} | {'Python (ms)':>12s} | {'CM7 est (ms)':>12s}")
    print("-" * 48)
    total_py = 0
    total_cm7 = 0
    cm7_factor = 0.03  # Python→C on CM7 ~30x speedup
    for stage, t_list in times.items():
        avg_ms = np.mean(t_list) * 1000
        cm7_ms = avg_ms * cm7_factor
        total_py += avg_ms
        total_cm7 += cm7_ms
        print(f"  {stage:>13s} | {avg_ms:>10.3f}ms | {cm7_ms:>10.3f}ms")
    print("-" * 48)
    print(f"  {'TOTAL':>13s} | {total_py:>10.3f}ms | {total_cm7:>10.3f}ms")
    print(f"\n  Scan rate capacity: {1000/total_cm7:.0f} scans/sec @ Cortex-M7 480MHz")


if __name__ == '__main__':
    np.random.seed(42)
    print_memory_map()
    demo_static_underdetermined()
    demo_tracking_crossing()
    demo_capacity_limit()
    demo_latency_profiling()
