#!/usr/bin/env python3
"""Debug improved SD-COP."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from iron_dome_sim.signal_model.array import UniformLinearArray
from iron_dome_sim.signal_model.signal_generator import generate_snapshots
from iron_dome_sim.signal_model.cumulant import compute_cumulant_matrix
from iron_dome_sim.doa import SequentialDeflationCOP

M, K = 8, 16
array = UniformLinearArray(M=M, d=0.5)
true_doas = np.radians(np.linspace(-65, 65, K))
scan_angles = np.linspace(-np.pi/2, np.pi/2, 3601)

np.random.seed(42)
X, _, _ = generate_snapshots(array, true_doas, 20, 1024, "non_stationary")

C = compute_cumulant_matrix(X, 2)
print("C shape:", C.shape)
eig = np.sort(np.abs(np.linalg.eigvalsh(C)))[::-1]
print("Eigenvalues:", np.round(eig[:5], 2), "...", np.round(eig[-3:], 4))

sdcop = SequentialDeflationCOP(array, rho=2, num_sources=K,
                                max_stages=8, spectrum_type="combined")
print("Batch_size:", sdcop.batch_size)

est, _ = sdcop.estimate(X, scan_angles)
print("Detected:", len(est))
for s in sdcop.stage_results:
    print("  Stage {}: {} det, energy_ratio={:.4f}".format(
        s['stage'], s['n_detected'], s['energy_ratio']))
    print("    DOAs (deg):", np.round(np.degrees(s['doas']), 1))

# Check detection accuracy
n_ok = sum(1 for d in true_doas if len(est) > 0 and min(abs(est - d)) < np.radians(3.0))
print("\nCorrect detections: {}/{}".format(n_ok, K))
print("Detected DOAs:", np.round(np.degrees(est), 1))
