#!/usr/bin/env python3
"""Iron Dome Full Simulation Demo.

Demonstrates underdetermined DOA estimation + multi-target tracking
in a realistic missile defense scenario.

Compares the proposed 2rho-th order Subspace COP algorithm against
MUSIC, ESPRIT, Capon, L1-SVD, and LASSO baselines.

Usage:
    python examples/demo_iron_dome.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from iron_dome_sim.signal_model.array import UniformLinearArray
from iron_dome_sim.signal_model.signal_generator import generate_snapshots
from iron_dome_sim.doa import SubspaceCOP, MUSIC, ESPRIT, Capon, L1SVD, LASSO_DOA
from iron_dome_sim.doa.spectrum import find_peaks_doa
from iron_dome_sim.tracking import MultiTargetTracker
from iron_dome_sim.tracking.state_models import ConstantVelocity
from iron_dome_sim.scenario.scenarios import iron_dome_scenario, small_scenario
from iron_dome_sim.eval.metrics import rmse_doa, detection_rate, crlb_doa
from iron_dome_sim.viz.plot_spectrum import plot_spectrum_comparison
from iron_dome_sim.viz.plot_metrics import plot_rmse_vs_snr, plot_comparison_bar
from iron_dome_sim.viz.plot_3d import plot_iron_dome_3d, plot_trajectories


def demo_doa_comparison():
    """Compare DOA estimators in underdetermined scenario."""
    print("=" * 60)
    print("Demo 1: DOA Estimation - Underdetermined Scenario")
    print("=" * 60)

    # Setup: 8 sensors, 15 sources (underdetermined!)
    M = 8
    K = 15
    array = UniformLinearArray(M=M, d=0.5)

    # True DOA angles (spread across -60 to 60 degrees)
    true_doas = np.radians(np.linspace(-55, 55, K))

    print(f"  Sensors: {M}")
    print(f"  Sources: {K} (underdetermined: K > M)")
    print(f"  COP max sources (rho=2): {array.max_sources(rho=2)}")
    print()

    # Generate data
    snr_db = 15
    T = 512
    np.random.seed(42)
    X, _, _ = generate_snapshots(array, true_doas, snr_db, T, "non_stationary")

    # DOA estimators
    cop = SubspaceCOP(array, rho=2, num_sources=K, spectrum_type="combined")
    music = MUSIC(array, num_sources=min(K, M - 1))
    capon = Capon(array, num_sources=K)
    l1svd = L1SVD(array, num_sources=K)
    lasso = LASSO_DOA(array, num_sources=K)

    scan_angles = np.linspace(-np.pi / 2, np.pi / 2, 1801)

    # Estimate DOA with each algorithm
    algorithms = {
        'Proposed (COP-4th)': cop,
        'MUSIC': music,
        'Capon/MVDR': capon,
        'L1-SVD': l1svd,
        'LASSO': lasso,
    }

    spectra = {}
    results_summary = {}

    print("Results:")
    print("-" * 50)

    for name, alg in algorithms.items():
        try:
            doa_est, spectrum = alg.estimate(X, scan_angles)
            rmse_val, _ = rmse_doa(doa_est, true_doas)
            pd, pfa = detection_rate(doa_est, true_doas)

            spectra[name] = spectrum
            results_summary[name] = {
                'rmse_deg': np.degrees(rmse_val),
                'n_detected': len(doa_est),
                'pd': pd,
            }

            print(f"  {name:20s}: RMSE={np.degrees(rmse_val):6.2f} deg, "
                  f"Detected={len(doa_est):2d}/{K}, Pd={pd:.2f}")
        except Exception as e:
            print(f"  {name:20s}: FAILED - {e}")
            spectra[name] = None
            results_summary[name] = {'rmse_deg': float('inf'), 'n_detected': 0, 'pd': 0}

    # CRLB
    crlb_var = crlb_doa(M, K, snr_db, T, rho=2)
    print(f"  {'CRLB (rho=2)':20s}: {np.degrees(np.sqrt(crlb_var[0])):6.2f} deg")

    # Visualize
    plot_spectrum_comparison(spectra, scan_angles, true_doas,
                             title=f'DOA Estimation: {K} sources, {M} sensors (SNR={snr_db}dB)',
                             save_path='doa_spectrum_comparison.png')

    # Bar chart comparison
    rmse_dict = {k: v['rmse_deg'] for k, v in results_summary.items()
                 if np.isfinite(v['rmse_deg'])}
    if rmse_dict:
        plot_comparison_bar(rmse_dict, metric_name="RMSE (degrees)",
                            title="DOA Estimation RMSE Comparison",
                            save_path='doa_rmse_bar.png')

    return results_summary


def demo_rmse_vs_snr():
    """RMSE vs SNR curves with CRLB."""
    print("\n" + "=" * 60)
    print("Demo 2: RMSE vs SNR (Monte Carlo)")
    print("=" * 60)

    M = 8
    K = 12
    array = UniformLinearArray(M=M, d=0.5)
    true_doas = np.radians(np.linspace(-50, 50, K))
    snr_range = np.arange(-5, 25, 3)
    T = 256
    n_trials = 20  # Reduced for speed

    estimators = [
        SubspaceCOP(array, rho=2, num_sources=K, spectrum_type="combined"),
        MUSIC(array, num_sources=min(K, M - 1)),
        Capon(array, num_sources=K),
        L1SVD(array, num_sources=K),
    ]

    results = {}
    for est in estimators:
        results[est.name] = {'rmse': [], 'snr': snr_range.tolist()}

    crlb_curve = []

    for snr in snr_range:
        print(f"  SNR = {snr} dB")
        crlb_var = crlb_doa(M, K, snr, T, rho=2)
        crlb_curve.append(np.sqrt(np.mean(crlb_var)))

        for est in estimators:
            rmse_trials = []
            for trial in range(n_trials):
                np.random.seed(42 + trial + int(snr * 100))
                X, _, _ = generate_snapshots(array, true_doas, snr, T,
                                             "non_stationary")
                try:
                    doa_est, _ = est.estimate(X)
                    r, _ = rmse_doa(doa_est, true_doas)
                    rmse_trials.append(r)
                except Exception:
                    rmse_trials.append(float('inf'))

            finite = [r for r in rmse_trials if np.isfinite(r)]
            avg = np.mean(finite) if finite else float('inf')
            results[est.name]['rmse'].append(avg)

    results['CRLB'] = {'rmse': crlb_curve, 'snr': snr_range.tolist()}

    plot_rmse_vs_snr(results,
                     title=f'RMSE vs SNR ({K} sources, {M} sensors)',
                     save_path='rmse_vs_snr.png')

    return results


def demo_iron_dome_scenario():
    """Full Iron Dome simulation with tracking."""
    print("\n" + "=" * 60)
    print("Demo 3: Iron Dome Full Scenario Simulation")
    print("=" * 60)

    # Use small scenario for faster demo
    scenario = small_scenario()
    network = scenario['network']
    threats = scenario['threats']
    threat_gen = scenario['threat_gen']
    interceptor = scenario['interceptor']
    dt = scenario['dt']
    duration = scenario['duration']

    print(f"  Scenario: {scenario['name']}")
    print(f"  Threats: {len(threats)}")
    print(f"  Radars: {len(network.sites)}")
    print(f"  Sensors per radar: {network.sites[0].array.M}")

    # Visualize threats
    plot_trajectories(threats,
                      title=f"Missile Trajectories - {scenario['name']}",
                      save_path='missile_trajectories.png')

    # Setup tracker with COP DOA estimator
    array = network.sites[0].array
    cop_estimator = SubspaceCOP(array, rho=2, num_sources=None,
                                spectrum_type="combined")

    def model_factory():
        return ConstantVelocity(dt=dt * 10, process_noise_std=0.01)

    from iron_dome_sim.tracking.multi_target_tracker import MultiTargetTracker
    tracker = MultiTargetTracker(
        doa_estimator=cop_estimator,
        model_factory=model_factory,
        filter_type="ekf",
        association_type="gnn",
        confirm_M=3,
        confirm_N=5,
        max_miss=5,
    )

    # Run simulation
    print("\n  Running simulation...")
    n_scans = int(duration / (dt * 10))
    confirmed_over_time = []
    doa_over_time = []

    for scan_idx in range(n_scans):
        t = scan_idx * dt * 10
        positions, active_ids = threat_gen.get_positions_at_time(threats, t)

        if len(positions) == 0:
            confirmed_over_time.append(0)
            continue

        # Generate radar data
        radar_data = network.generate_all_snapshots(positions, T=64,
                                                     signal_type="missile")

        # Process first radar with good data
        processed = False
        for rd in radar_data:
            if rd['X'] is not None and rd['n_targets'] > 0:
                confirmed, doa_est, _ = tracker.process_scan(rd['X'])
                confirmed_over_time.append(len(confirmed))
                doa_over_time.append(doa_est)
                processed = True
                break

        if not processed:
            confirmed_over_time.append(0)

        if scan_idx % 10 == 0:
            print(f"    Scan {scan_idx:3d}/{n_scans}: "
                  f"Active threats={len(active_ids)}, "
                  f"Confirmed tracks={confirmed_over_time[-1]}")

    # Plot tracking results
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(np.arange(len(confirmed_over_time)) * dt * 10,
            confirmed_over_time, 'b-', linewidth=2, label='Confirmed Tracks')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Number of Tracks')
    ax.set_title(f"Tracking Performance - {scenario['name']}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('tracking_performance.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 3D visualization
    track_histories = tracker.get_track_histories()
    plot_iron_dome_3d(
        threats, network.sites, interceptor,
        title=f"Iron Dome 3D - {scenario['name']}",
        save_path='iron_dome_3d.png',
    )

    # Interception statistics
    intercept_stats = interceptor.get_statistics()
    print(f"\n  Interception Stats: {intercept_stats}")

    return tracker


if __name__ == "__main__":
    print("Iron Dome DOA Estimation + Tracking Simulation")
    print("Based on: Choi & Yoo, IEEE TSP 2015")
    print("Subspace Constrained Optimization + Multi-Target Tracking")
    print()

    # Run demos
    demo_doa_comparison()
    demo_rmse_vs_snr()
    demo_iron_dome_scenario()

    print("\n" + "=" * 60)
    print("All demos completed. Figures saved to current directory.")
    print("=" * 60)
