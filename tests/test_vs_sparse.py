#!/usr/bin/env python3
"""Compare COP family vs underdetermined algorithms (L1-SVD, LASSO).
K=16, M=8, T=1024, SNR sweep
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from iron_dome_sim.signal_model.array import UniformLinearArray
from iron_dome_sim.signal_model.signal_generator import generate_snapshots
from iron_dome_sim.doa import SubspaceCOP, TemporalCOP, SequentialDeflationCOP
from iron_dome_sim.doa.sparse_recovery import L1SVD, LASSO_DOA

M, K, T = 8, 16, 1024
array = UniformLinearArray(M=M, d=0.5)
true_doas = np.radians(np.linspace(-65, 65, K))
scan_angles = np.linspace(-np.pi/2, np.pi/2, 3601)

snr_values = [10, 15, 20, 25, 30]

def count_correct(est, true, thr_deg=3.0):
    if len(est) == 0:
        return 0
    return sum(1 for d in true if min(abs(est - d)) < np.radians(thr_deg))

print("=" * 90)
print(f"COP Family vs Sparse Recovery  |  M={M}, K={K}, T={T}")
print("=" * 90)
print(f"{'SNR':>5}  {'COP':>6}  {'T-COP':>6}  {'SD-COP':>7}  {'L1-SVD':>7}  {'LASSO':>6}")
print("-" * 90)

for snr in snr_values:
    np.random.seed(42)
    X, _, _ = generate_snapshots(array, true_doas, snr, T, "non_stationary")

    # COP
    cop = SubspaceCOP(array, rho=2, num_sources=min(K, 14), spectrum_type="combined")
    cop_est, _ = cop.estimate(X, scan_angles)
    cop_ok = count_correct(cop_est, true_doas)

    # T-COP (10 scans)
    tcop = TemporalCOP(array, rho=2, num_sources=min(K, 14), alpha=0.85, prior_weight=0.0)
    for s in range(10):
        np.random.seed(42 + s)
        Xs, _, _ = generate_snapshots(array, true_doas, snr, T, "non_stationary")
        tcop.spectrum(Xs, scan_angles)
    tcop_est, _ = tcop.estimate(Xs, scan_angles)
    tcop_ok = count_correct(tcop_est, true_doas)

    # SD-COP
    sdcop = SequentialDeflationCOP(array, rho=2, num_sources=K,
                                    max_stages=8, spectrum_type="combined")
    sdcop_est, _ = sdcop.estimate(X, scan_angles)
    sdcop_ok = count_correct(sdcop_est, true_doas)

    # L1-SVD
    try:
        l1 = L1SVD(array, num_sources=K, grid_size=len(scan_angles), reg_param=0.1)
        l1_est, _ = l1.estimate(X, scan_angles)
        l1_ok = count_correct(l1_est, true_doas)
    except Exception as e:
        l1_ok = -1
        l1_est = []

    # LASSO
    try:
        lasso = LASSO_DOA(array, num_sources=K, grid_size=len(scan_angles), reg_param=0.05)
        lasso_est, _ = lasso.estimate(X, scan_angles)
        lasso_ok = count_correct(lasso_est, true_doas)
    except Exception as e:
        lasso_ok = -1
        lasso_est = []

    l1_str = f"{l1_ok:>2}/{len(l1_est)}" if l1_ok >= 0 else "FAIL"
    la_str = f"{lasso_ok:>2}/{len(lasso_est)}" if lasso_ok >= 0 else "FAIL"

    print(f"{snr:>4}dB  {cop_ok:>2}/{len(cop_est):<3}  {tcop_ok:>2}/{len(tcop_est):<3}  "
          f"{sdcop_ok:>2}/{len(sdcop_est):<4}  {l1_str:>7}  {la_str:>6}")

print("=" * 90)
print("Format: correct_detections / total_peaks  (threshold: 3°)")
