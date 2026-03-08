"""Monte Carlo simulation runner for statistical evaluation.

Runs multiple independent trials and computes statistics
for DOA estimation and tracking performance.
"""

import numpy as np
from .metrics import rmse_doa, gospa, detection_rate, crlb_doa


class MonteCarloRunner:
    """Monte Carlo simulation for DOA + Tracking evaluation.

    Args:
        n_trials: Number of independent trials.
        seed: Random seed for reproducibility.
    """

    def __init__(self, n_trials=100, seed=42):
        self.n_trials = n_trials
        self.seed = seed

    def evaluate_doa(self, estimators, array, true_doas, snr_range,
                     T=256, signal_type="non_stationary"):
        """Evaluate DOA estimators over SNR range.

        Args:
            estimators: List of DOA estimator objects.
            array: Array geometry object.
            true_doas: True DOA angles in radians.
            snr_range: List of SNR values in dB.
            T: Number of snapshots per trial.
            signal_type: Signal type string.

        Returns:
            results: Dict with RMSE curves for each estimator.
        """
        from ..signal_model.signal_generator import generate_snapshots

        results = {est.name: {'rmse': [], 'pd': [], 'snr': snr_range}
                   for est in estimators}

        # Add CRLB
        K = len(true_doas)
        crlb_curve = []

        for snr in snr_range:
            print(f"  SNR = {snr} dB")

            # CRLB at this SNR
            crlb_var = crlb_doa(array.M, K, snr, T)
            crlb_curve.append(np.sqrt(np.mean(crlb_var)))

            for est in estimators:
                rmse_trials = []
                pd_trials = []

                for trial in range(self.n_trials):
                    np.random.seed(self.seed + trial + int(snr * 100))

                    # Generate data
                    X, _, _ = generate_snapshots(array, true_doas, snr, T,
                                                 signal_type)

                    # Estimate DOA
                    try:
                        doa_est, _ = est.estimate(X)
                        rmse_val, _ = rmse_doa(doa_est, true_doas)
                        pd_val, _ = detection_rate(doa_est, true_doas)
                    except Exception:
                        rmse_val = float('inf')
                        pd_val = 0.0

                    rmse_trials.append(rmse_val)
                    pd_trials.append(pd_val)

                # Average over trials (exclude inf)
                finite_rmse = [r for r in rmse_trials if np.isfinite(r)]
                avg_rmse = np.mean(finite_rmse) if finite_rmse else float('inf')
                avg_pd = np.mean(pd_trials)

                results[est.name]['rmse'].append(avg_rmse)
                results[est.name]['pd'].append(avg_pd)

        results['CRLB'] = {'rmse': crlb_curve, 'snr': snr_range}

        return results

    def evaluate_tracking(self, tracker_configs, scenario_factory, n_scans=100):
        """Evaluate tracking performance with different DOA estimators.

        Args:
            tracker_configs: List of (name, MultiTargetTracker) tuples.
            scenario_factory: Callable returning scenario dict.
            n_scans: Number of radar scans to simulate.

        Returns:
            results: Dict with GOSPA and track metrics per tracker.
        """
        results = {}

        for name, tracker in tracker_configs:
            print(f"  Evaluating: {name}")
            gospa_per_scan = []
            n_confirmed_per_scan = []

            for trial in range(self.n_trials):
                np.random.seed(self.seed + trial)
                tracker.reset()

                scenario = scenario_factory()
                network = scenario['network']
                threats = scenario['threats']
                threat_gen = scenario['threat_gen']
                dt = scenario['dt']

                trial_gospa = []
                trial_confirmed = []

                for scan in range(n_scans):
                    t = scan * dt * 10  # scan interval

                    # Get current threat positions
                    positions, active_ids = threat_gen.get_positions_at_time(
                        threats, t
                    )

                    if len(positions) == 0:
                        continue

                    # Generate radar data from first radar
                    radar_data = network.generate_all_snapshots(
                        positions, T=64, signal_type="missile"
                    )

                    # Process first radar with tracker
                    for rd in radar_data:
                        if rd['X'] is not None:
                            confirmed, doa_est, _ = tracker.process_scan(rd['X'])
                            break
                    else:
                        continue

                    # Evaluate GOSPA (using DOA angles as positions)
                    est_doas = np.array([t.filter.x[:2] for t in confirmed]) \
                        if confirmed else np.empty((0, 2))
                    true_doas = np.column_stack([
                        rd['true_az'], rd['true_el']
                    ]) if rd['n_targets'] > 0 else np.empty((0, 2))

                    g_val, _ = gospa(est_doas, true_doas, c=0.5)
                    trial_gospa.append(g_val)
                    trial_confirmed.append(len(confirmed))

                gospa_per_scan.append(trial_gospa)
                n_confirmed_per_scan.append(trial_confirmed)

            # Average over trials
            if gospa_per_scan:
                max_len = max(len(g) for g in gospa_per_scan)
                padded = np.full((len(gospa_per_scan), max_len), np.nan)
                for i, g in enumerate(gospa_per_scan):
                    padded[i, :len(g)] = g
                avg_gospa = np.nanmean(padded, axis=0)
            else:
                avg_gospa = []

            results[name] = {
                'gospa': avg_gospa,
                'n_confirmed': n_confirmed_per_scan,
            }

        return results
