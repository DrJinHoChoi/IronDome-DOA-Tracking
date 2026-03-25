# -*- coding: utf-8 -*-
"""COP-RFS DOA Simulation: 10 scans with moving targets."""
import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
os.environ['PYTHONIOENCODING'] = 'utf-8'

import numpy as np
import time
from scipy.linalg import toeplitz

W = 70
NUM_ANGLES = 181
scan_deg = np.linspace(-90, 90, NUM_ANGLES)
scan_rad = np.radians(scan_deg)

def gen_signal(M, T, doas_deg, snr=20):
    K = len(doas_deg)
    A = np.zeros((M, K), dtype=complex)
    for k in range(K):
        for m in range(M):
            A[m, k] = np.exp(1j * np.pi * m * np.sin(np.radians(doas_deg[k])))
    S = (2*(np.random.randint(0,2,(K,T)))-1).astype(complex) * np.sqrt(10**(snr/10))
    return A @ S + (np.random.randn(M,T)+1j*np.random.randn(M,T))/np.sqrt(2)

def cop_spectrum(X, M_v, K):
    M, T = X.shape
    L = 2*(M-1)
    R = X @ X.conj().T / T
    c4 = np.zeros(M_v, dtype=complex)
    cnt = np.zeros(M_v)
    for i1 in range(M):
        for i2 in range(M):
            xi12 = X[i1]*X[i2]
            for i3 in range(M):
                xi3c = X[i3].conj()
                for i4 in range(M):
                    tau = (i1+i2)-(i3+i4)
                    if 0 <= tau <= L:
                        m4 = np.mean(xi12 * xi3c * X[i4].conj())
                        c4[tau] += m4 - R[i1,i3]*R[i2,i4] - R[i1,i4]*R[i2,i3]
                        cnt[tau] += 1
    mask = cnt > 0
    c4[mask] /= cnt[mask]
    C = toeplitz(c4, c4.conj())
    eigvals, eigvecs = np.linalg.eigh(C)
    idx = np.argsort(np.abs(eigvals))[::-1]
    eigvecs = eigvecs[:, idx]
    K = min(K, M_v-1)
    U_s, U_n = eigvecs[:,:K], eigvecs[:,K:]
    spec = np.zeros(NUM_ANGLES)
    for i, th in enumerate(scan_rad):
        a_v = np.exp(1j*np.pi*np.arange(M_v)*np.sin(th))
        num = np.real(np.sum(np.abs(U_s.conj().T @ a_v)**2))
        den = np.real(np.sum(np.abs(U_n.conj().T @ a_v)**2))
        spec[i] = num / max(den, 1e-15)
    mx = np.max(spec)
    if mx > 0: spec /= mx
    return 10*np.log10(spec + 1e-10)

def find_peaks(spec_db, K, thr=-12):
    peaks = []
    for i in range(2, NUM_ANGLES-2):
        if spec_db[i] > spec_db[i-1] and spec_db[i] > spec_db[i+1] and spec_db[i] > thr:
            peaks.append((spec_db[i], scan_deg[i]))
    peaks.sort(reverse=True)
    doas = sorted([p[1] for p in peaks[:K]])
    filt = []
    for d in doas:
        if not filt or abs(d - filt[-1]) > 3:
            filt.append(d)
    return filt

def bar_chart(spec_db, doas, label):
    idx = np.linspace(0, NUM_ANGLES-1, W).astype(int)
    vals = spec_db[idx]
    norm = np.clip((vals + 30) / 30, 0, 1)
    levels = (norm * 8).astype(int)
    chars = [' ', '.', ':', '|', '#', '#', 'X', 'X', 'X']
    bar = [chars[min(l,8)] for l in levels]
    for doa in doas:
        pos = int((doa+90)/180*(W-1))
        pos = max(0, min(W-1, pos))
        bar[pos] = 'V'
    return f" {label:>5s} |{''.join(bar)}|"

def dot_line(doas):
    line = list('-' * W)
    c = W // 2
    line[c] = '|'
    for doa in doas:
        pos = int(c + doa * c / 90)
        pos = max(0, min(W-1, pos))
        line[pos] = 'O'
    return f"       |{''.join(line)}|"

M = 4
T = 128
M_v = 2*(M-1)+1

scenarios = [
    ([-40, 20],       "Scan 1: Two sources at -40, +20"),
    ([-30, 20],       "Scan 2: Target 1 moves -40 -> -30"),
    ([-20, 15],       "Scan 3: Converging: -20, +15"),
    ([-10, 10],       "Scan 4: Close: -10, +10"),
    ([-5, 5],         "Scan 5: Very close: -5, +5"),
    ([0, 0],          "Scan 6: CROSSING at 0 deg!"),
    ([5, -5],         "Scan 7: Post-crossing (swapped)"),
    ([15, -15, 40],   "Scan 8: New target at +40!"),
    ([25, -25, 40],   "Scan 9: 3 targets diverging"),
    ([35, -35, 40],   "Scan 10: 3 targets separated"),
]

np.random.seed(42)

for i, (doas, desc) in enumerate(scenarios):
    doas = np.array(doas, dtype=float)
    K = len(doas)
    X = gen_signal(M, T, doas, snr=20)

    t0 = time.perf_counter()
    spec_db = cop_spectrum(X, M_v, K)
    est = find_peaks(spec_db, K)
    ms = (time.perf_counter() - t0) * 1000

    print()
    print(f"  {'='*76}")
    print(f"  {desc}")
    print(f"  {'='*76}")
    print(f"   deg  |-90       -60        -30       0          +30       +60        +90|")
    print(f"  {'-'*78}")
    print(bar_chart(spec_db, est, "COP"))
    print(dot_line(est))
    print(f"  {'-'*78}")

    true_s = ", ".join(f"{d:+.0f}" for d in doas)
    est_s = ", ".join(f"{d:+.0f}" for d in est) if est else "---"
    print(f"  True : {true_s} deg")
    print(f"  COP  : {est_s} deg")

    if est:
        errs = [min(abs(e - t) for t in doas) for e in est]
        rmse = np.sqrt(np.mean(np.array(errs)**2))
        print(f"  RMSE : {rmse:.1f} deg  |  Latency: {ms:.1f} ms")
    else:
        print(f"  RMSE : N/A  |  Latency: {ms:.1f} ms")

    print(f"  V = COP peak    O = DOA position")

print()
print(f"  {'='*76}")
print(f"  SIMULATION COMPLETE (10/10 scans)")
print(f"  {'='*76}")
print(f"  M=4 mics, rho=2, M_v=7 virtual array")
print(f"  COP capacity: 6 sources (MUSIC: only 3)")
print(f"  Demonstrated: target motion, crossing, birth")
print()
