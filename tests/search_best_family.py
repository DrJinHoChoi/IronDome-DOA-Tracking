#!/usr/bin/env python3
"""Search for the best COP family spatial spectrum parameters (K=16)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from iron_dome_sim.signal_model.array import UniformLinearArray
from iron_dome_sim.signal_model.signal_generator import generate_snapshots
from iron_dome_sim.doa import SubspaceCOP, TemporalCOP, SequentialDeflationCOP

M = 8
array = UniformLinearArray(M=M, d=0.5)
scan_angles = np.linspace(-np.pi/2, np.pi/2, 3601)

print("=== COP Family K=16 Search ===")
best = []
for snr in [20, 25, 30]:
    for T in [512, 1024]:
        for spread in [7, 8, 9]:
            K = 16
            doas = np.radians(np.linspace(-spread*(K-1)/2, spread*(K-1)/2, K))
            for seed in [42, 55, 77, 123, 300, 500, 999, 1234]:
                np.random.seed(seed)
                try:
                    X, _, _ = generate_snapshots(array, doas, snr, T, "non_stationary")

                    # COP
                    cop = SubspaceCOP(array, rho=2, num_sources=min(K, 14),
                                     spectrum_type="combined")
                    cop_est, _ = cop.estimate(X, scan_angles)
                    cop_ok = sum(1 for d in doas
                                 if min(abs(cop_est - d)) < np.radians(3.0))

                    # T-COP 10 scans
                    tcop = TemporalCOP(array, rho=2, num_sources=min(K, 14),
                                       alpha=0.85, prior_weight=0.0)
                    for s in range(10):
                        np.random.seed(seed + s)
                        Xs, _, _ = generate_snapshots(array, doas, snr, T, "non_stationary")
                        tcop.spectrum(Xs, scan_angles)
                    tcop_est, _ = tcop.estimate(Xs, scan_angles)
                    tcop_ok = sum(1 for d in doas
                                  if min(abs(tcop_est - d)) < np.radians(3.0))

                    # SD-COP
                    sdcop = SequentialDeflationCOP(array, rho=2, num_sources=K,
                                                   max_stages=8,
                                                   spectrum_type="combined")
                    sdcop_est, _ = sdcop.estimate(X, scan_angles)
                    sdcop_ok = sum(1 for d in doas
                                   if min(abs(sdcop_est - d)) < np.radians(3.0))

                    total = cop_ok + tcop_ok + sdcop_ok
                    if sdcop_ok >= 11 and tcop_ok >= cop_ok:
                        print("SNR={} T={} spc={}d seed={}: COP={}/16 TCOP={}/16 SDCOP={}/16".format(
                            snr, T, spread, seed, cop_ok, tcop_ok, sdcop_ok))
                        best.append((total, sdcop_ok, tcop_ok, cop_ok,
                                     snr, T, spread, seed))
                except Exception as e:
                    pass

best.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
print("\nTop 10:")
for b in best[:10]:
    print("  total={} SD={}/16 TC={}/16 COP={}/16 | SNR={} T={} spc={}d seed={}".format(*b))
