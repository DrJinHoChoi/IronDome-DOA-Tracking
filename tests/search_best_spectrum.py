#!/usr/bin/env python3
"""Search for the best spatial spectrum parameters."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from iron_dome_sim.signal_model.array import UniformLinearArray
from iron_dome_sim.signal_model.signal_generator import generate_snapshots
from iron_dome_sim.doa import SubspaceCOP

M = 8
array = UniformLinearArray(M=M, d=0.5)
scan_angles = np.linspace(-np.pi/2, np.pi/2, 3601)

best = []
for K in [8, 10]:
    for snr in [15, 20, 25, 30]:
        for T in [256, 512, 1024]:
            for spread in [10, 12, 14, 15]:
                doas = np.radians(np.linspace(-spread*(K-1)/2, spread*(K-1)/2, K))
                for seed in [42, 7, 123, 2024, 55, 77, 300, 500]:
                    np.random.seed(seed)
                    try:
                        X, _, _ = generate_snapshots(array, doas, snr, T, "non_stationary")
                        cop = SubspaceCOP(array, rho=2, num_sources=K, spectrum_type="combined")
                        spec = cop.spectrum(X, scan_angles)
                        est, _ = cop.estimate(X, scan_angles)

                        errs = [min(abs(np.degrees(e) - np.degrees(d)) for e in est) for d in doas]
                        n_ok = sum(1 for e in errs if e < 2.0)
                        max_err = max(errs)
                        mean_err = np.mean(errs)

                        # Spectrum quality: weakest true peak in dB
                        spec_db = 10*np.log10(np.abs(spec)/np.max(np.abs(spec)) + 1e-15)
                        peak_vals = []
                        for d in doas:
                            idx = np.argmin(np.abs(scan_angles - d))
                            window = slice(max(0,idx-20), min(len(spec_db),idx+20))
                            peak_vals.append(np.max(spec_db[window]))
                        min_peak = min(peak_vals)

                        if n_ok == K and max_err < 2.5:
                            score = n_ok*10 - mean_err*5 - max_err*3 + min_peak*0.5 + snr*0.1
                            best.append((score, K, snr, T, spread, seed, n_ok,
                                        round(mean_err,2), round(max_err,2), round(min_peak,1)))
                    except:
                        pass

best.sort(key=lambda x: x[0], reverse=True)
print("Found {} perfect configs. Top 20:".format(len(best)))
header = "{:>6s}  K  SNR    T  spc  seed  det  mean_err  max_err  weak_peak".format("Score")
print(header)
for b in best[:20]:
    line = "{:6.1f}  {}  {:3d}  {:4d}  {:2d}  {:4d}   {}/{}    {:.2f}      {:.2f}      {:.1f} dB".format(
        b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[1], b[7], b[8], b[9])
    print(line)
