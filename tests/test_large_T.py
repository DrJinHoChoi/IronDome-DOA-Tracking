#!/usr/bin/env python3
"""Test SD-COP performance vs number of snapshots T.

Experiment: K=16 (super-underdetermined), M=8, SNR=20dB
Sweep T = [256, 512, 1024, 2048, 4096, 8192]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from iron_dome_sim.signal_model.array import UniformLinearArray
from iron_dome_sim.signal_model.signal_generator import generate_snapshots
from iron_dome_sim.doa import SubspaceCOP, TemporalCOP, SequentialDeflationCOP

M = 8
K = 16
array = UniformLinearArray(M=M, d=0.5)
true_doas = np.radians(np.linspace(-65, 65, K))
scan_angles = np.linspace(-np.pi/2, np.pi/2, 3601)

T_values = [256, 512, 1024, 2048, 4096, 8192]

print("=" * 70)
print(f"SD-COP vs T (snapshots)  |  M={M}, K={K}, SNR=20dB")
print("=" * 70)
print(f"{'T':>6}  {'COP det':>8}  {'COP ok':>7}  {'TCOP det':>9}  {'TCOP ok':>8}  {'SDCOP det':>10}  {'SDCOP ok':>9}")
print("-" * 70)

for T in T_values:
    np.random.seed(42)
    X, _, _ = generate_snapshots(array, true_doas, 20, T, "non_stationary")

    # COP
    cop = SubspaceCOP(array, rho=2, num_sources=min(K, 14), spectrum_type="combined")
    cop_est, _ = cop.estimate(X, scan_angles)
    cop_ok = sum(1 for d in true_doas if len(cop_est) > 0 and min(abs(cop_est - d)) < np.radians(3.0))

    # T-COP (10 scans)
    tcop = TemporalCOP(array, rho=2, num_sources=min(K, 14), alpha=0.85, prior_weight=0.0)
    for s in range(10):
        np.random.seed(42 + s)
        Xs, _, _ = generate_snapshots(array, true_doas, 20, T, "non_stationary")
        tcop.spectrum(Xs, scan_angles)
    tcop_est, _ = tcop.estimate(Xs, scan_angles)
    tcop_ok = sum(1 for d in true_doas if len(tcop_est) > 0 and min(abs(tcop_est - d)) < np.radians(3.0))

    # SD-COP
    sdcop = SequentialDeflationCOP(array, rho=2, num_sources=K,
                                    max_stages=8, spectrum_type="combined")
    sdcop_est, _ = sdcop.estimate(X, scan_angles)
    sdcop_ok = sum(1 for d in true_doas if len(sdcop_est) > 0 and min(abs(sdcop_est - d)) < np.radians(3.0))

    print(f"{T:>6}  {len(cop_est):>8}  {cop_ok:>7}  {len(tcop_est):>9}  {tcop_ok:>8}  {len(sdcop_est):>10}  {sdcop_ok:>9}")

print("=" * 70)
print("det = # detected peaks, ok = # within 3° of true DOA")
