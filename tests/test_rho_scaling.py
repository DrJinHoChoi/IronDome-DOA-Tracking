#!/usr/bin/env python3
"""Test COP performance vs rho (cumulant order).

Experiment: Compare COP at rho=1 (2nd-order), rho=2 (4th-order), rho=3 (6th-order).
M=8 sensors, SNR=20 dB, T=1024 snapshots.

Virtual array size: M_v = rho*(M-1)+1
Max resolvable sources: K_max = rho*(M-1)

  rho=1 -> M_v= 8, K_max= 7
  rho=2 -> M_v=15, K_max=14
  rho=3 -> M_v=22, K_max=21
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from iron_dome_sim.signal_model.array import UniformLinearArray
from iron_dome_sim.signal_model.signal_generator import generate_snapshots
from iron_dome_sim.doa import SubspaceCOP

M, T, SNR = 8, 1024, 20
array = UniformLinearArray(M=M, d=0.5)
scan_angles = np.linspace(-np.pi / 2, np.pi / 2, 3601)

rho_values = [1, 2, 3]
K_values = [4, 7, 10, 14, 18, 21]
seeds = [42, 55, 77]
tol_deg = 3.0

# Capacity limits per rho
capacity = {rho: rho * (M - 1) for rho in rho_values}  # {1:7, 2:14, 3:21}

print("=" * 85)
print(f"COP Performance vs rho (cumulant order)  |  M={M}, SNR={SNR} dB, T={T}")
print(f"Seeds: {seeds}  |  Detection tolerance: {tol_deg} deg")
print("=" * 85)
print(f"  rho=1: M_v={1*(M-1)+1:>2}, K_max={capacity[1]:>2}")
print(f"  rho=2: M_v={2*(M-1)+1:>2}, K_max={capacity[2]:>2}")
print(f"  rho=3: M_v={3*(M-1)+1:>2}, K_max={capacity[3]:>2}")
print("=" * 85)

# Storage: results[rho][K] = list of (det_count, rmse) per seed
results = {rho: {} for rho in rho_values}

for K in K_values:
    true_doas = np.radians(np.linspace(-65, 65, K))

    for rho in rho_values:
        det_list = []
        rmse_list = []

        # num_sources: limited by capacity (need at least 1 noise eigenvector)
        ns = min(K, capacity[rho])

        for seed in seeds:
            np.random.seed(seed)
            X, _, _ = generate_snapshots(array, true_doas, SNR, T, "non_stationary")

            cop = SubspaceCOP(array, rho=rho, num_sources=ns, spectrum_type="combined")
            est, _ = cop.estimate(X, scan_angles)

            # Count correct detections (within tolerance)
            ok = 0
            errs = []
            for d in true_doas:
                if len(est) > 0:
                    min_err = np.min(np.abs(est - d))
                    if min_err < np.radians(tol_deg):
                        ok += 1
                        errs.append(np.degrees(min_err))
            det_list.append(ok)
            rmse = np.sqrt(np.mean(np.array(errs) ** 2)) if errs else float('nan')
            rmse_list.append(rmse)

        avg_det = np.mean(det_list)
        avg_rmse = np.nanmean(rmse_list)
        results[rho][K] = (avg_det, avg_rmse, det_list)

# Print detection table
print()
print(f"{'K':>4}  {'rho=1 det':>10} {'rho=1 RMSE':>11}  {'rho=2 det':>10} {'rho=2 RMSE':>11}  {'rho=3 det':>10} {'rho=3 RMSE':>11}")
print("-" * 85)

for K in K_values:
    cols = []
    for rho in rho_values:
        avg_det, avg_rmse, det_list = results[rho][K]
        cap = capacity[rho]
        marker = "*" if K > cap else " "
        cols.append(f"{avg_det:>7.1f}/{K:<2}{marker} {avg_rmse:>9.3f}d")
    print(f"{K:>4}  {'  '.join(cols)}")

print("-" * 85)
print("  * = K exceeds capacity for that rho")

# Per-seed detail
print()
print("=" * 85)
print("Per-seed detection counts")
print("=" * 85)
for K in K_values:
    print(f"\n  K={K} sources:")
    for rho in rho_values:
        _, _, det_list = results[rho][K]
        det_str = ", ".join(f"{d}" for d in det_list)
        print(f"    rho={rho}: [{det_str}]  avg={np.mean(det_list):.1f}/{K}")

# Summary
print()
print("=" * 85)
print("Summary: Average detection rate (%) by rho")
print("=" * 85)
print(f"{'K':>4}  {'rho=1':>8}  {'rho=2':>8}  {'rho=3':>8}")
print("-" * 40)
for K in K_values:
    rates = []
    for rho in rho_values:
        avg_det, _, _ = results[rho][K]
        rates.append(f"{100*avg_det/K:>7.1f}%")
    print(f"{K:>4}  {'  '.join(rates)}")
print("=" * 85)
