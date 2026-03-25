# -*- coding: utf-8 -*-
"""DOA Estimation Comparison: CBF vs MVDR vs MUSIC vs COP + GM-PHD Tracking.

10-scan simulation with 4 methods side-by-side.
Demonstrates resolution hierarchy: CBF < MVDR < MUSIC < COP.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import time
from scipy.linalg import toeplitz

W = 68
N_ANGLES = 181
scan_deg = np.linspace(-90, 90, N_ANGLES)
scan_rad = np.radians(scan_deg)
M = 4
M_V = 2 * (M - 1) + 1  # 7


def gen_signal(M, T, doas_deg, snr=20):
    K = len(doas_deg)
    A = np.zeros((M, K), dtype=complex)
    for k in range(K):
        for m in range(M):
            A[m, k] = np.exp(1j * np.pi * m * np.sin(np.radians(doas_deg[k])))
    S = (2 * (np.random.randint(0, 2, (K, T))) - 1).astype(complex) * np.sqrt(10 ** (snr / 10))
    return A @ S + (np.random.randn(M, T) + 1j * np.random.randn(M, T)) / np.sqrt(2)


def compute_cumulant(X):
    """Compute 4th-order cumulant matrix (shared by COP methods)."""
    T = X.shape[1]
    L = 2 * (M - 1)
    R = X @ X.conj().T / T
    c4 = np.zeros(M_V, dtype=complex)
    cnt = np.zeros(M_V)
    for i1 in range(M):
        for i2 in range(M):
            xi12 = X[i1] * X[i2]
            for i3 in range(M):
                xi3c = X[i3].conj()
                for i4 in range(M):
                    tau = (i1 + i2) - (i3 + i4)
                    if 0 <= tau <= L:
                        m4 = np.mean(xi12 * xi3c * X[i4].conj())
                        c4[tau] += m4 - R[i1, i3] * R[i2, i4] - R[i1, i4] * R[i2, i3]
                        cnt[tau] += 1
    mask = cnt > 0
    c4[mask] /= cnt[mask]
    return toeplitz(c4, c4.conj())


# --- Physical array methods (M=4) ---

def cbf_spec(X):
    """Conventional Beamforming: P = a^H R a  (no K needed)"""
    R = X @ X.conj().T / X.shape[1]
    spec = np.zeros(N_ANGLES)
    for i in range(N_ANGLES):
        a = np.exp(1j * np.pi * np.arange(M) * np.sin(scan_rad[i]))
        spec[i] = np.real(a.conj() @ R @ a)
    mx = np.max(spec)
    if mx > 0: spec /= mx
    return 10 * np.log10(spec + 1e-10)


def mvdr_spec(X):
    """MVDR (Capon): P = 1 / (a^H R^{-1} a)  (no K needed)"""
    R = X @ X.conj().T / X.shape[1] + 1e-6 * np.eye(M)
    R_inv = np.linalg.inv(R)
    spec = np.zeros(N_ANGLES)
    for i in range(N_ANGLES):
        a = np.exp(1j * np.pi * np.arange(M) * np.sin(scan_rad[i]))
        spec[i] = 1.0 / max(np.real(a.conj() @ R_inv @ a), 1e-15)
    mx = np.max(spec)
    if mx > 0: spec /= mx
    return 10 * np.log10(spec + 1e-10)


def music_spec(X, K):
    """MUSIC: P = 1 / (a^H U_n U_n^H a)  (K required)"""
    R = X @ X.conj().T / X.shape[1]
    eigvals, eigvecs = np.linalg.eigh(R)
    idx = np.argsort(np.abs(eigvals))[::-1]
    U_n = eigvecs[:, idx][:, min(K, M-1):]
    spec = np.zeros(N_ANGLES)
    for i in range(N_ANGLES):
        a = np.exp(1j * np.pi * np.arange(M) * np.sin(scan_rad[i]))
        den = np.real(np.sum(np.abs(U_n.conj().T @ a) ** 2))
        spec[i] = 1.0 / max(den, 1e-15)
    mx = np.max(spec)
    if mx > 0: spec /= mx
    return 10 * np.log10(spec + 1e-10)


# --- COP Virtual array methods (M_v=7, no K needed for beamforming) ---

def cop_cbf_spec(C):
    """COP-CBF: Conventional Beamforming on virtual array cumulant.

    P = a_v^H C a_v   (NO K needed)
    Uses M_v=7 virtual steering vector on cumulant matrix.
    Better resolution than physical CBF (M_v > M).
    Gaussian noise automatically eliminated.
    """
    spec = np.zeros(N_ANGLES)
    for i in range(N_ANGLES):
        a_v = np.exp(1j * np.pi * np.arange(M_V) * np.sin(scan_rad[i]))
        spec[i] = np.real(a_v.conj() @ C @ a_v)
    # Handle negative values (cumulant can be negative)
    spec = np.abs(spec)
    mx = np.max(spec)
    if mx > 0: spec /= mx
    return 10 * np.log10(spec + 1e-10)


def cop_mvdr_spec(C):
    """COP-MVDR: Capon beamforming on virtual array cumulant.

    P = 1 / (a_v^H C^{-1} a_v)   (NO K needed)
    MVDR on M_v=7 cumulant matrix.
    Sharper peaks than COP-CBF, no source count needed.
    Noise-free cumulant -> no diagonal loading needed in theory.
    """
    C_reg = C + 1e-6 * np.eye(M_V)
    try:
        C_inv = np.linalg.inv(C_reg)
    except np.linalg.LinAlgError:
        C_inv = np.linalg.inv(C + 1e-4 * np.eye(M_V))

    spec = np.zeros(N_ANGLES)
    for i in range(N_ANGLES):
        a_v = np.exp(1j * np.pi * np.arange(M_V) * np.sin(scan_rad[i]))
        val = np.real(a_v.conj() @ C_inv @ a_v)
        spec[i] = 1.0 / max(abs(val), 1e-15)
    mx = np.max(spec)
    if mx > 0: spec /= mx
    return 10 * np.log10(spec + 1e-10)


def cop_spec(C, K):
    """COP subspace: P = (a_v^H P_s a_v) / (a_v^H P_n a_v)  (K required)"""
    eigvals, eigvecs = np.linalg.eigh(C)
    idx = np.argsort(np.abs(eigvals))[::-1]
    eigvecs = eigvecs[:, idx]
    K = min(K, M_V - 1)
    U_s, U_n = eigvecs[:, :K], eigvecs[:, K:]
    spec = np.zeros(N_ANGLES)
    for i in range(N_ANGLES):
        a_v = np.exp(1j * np.pi * np.arange(M_V) * np.sin(scan_rad[i]))
        num = np.real(np.sum(np.abs(U_s.conj().T @ a_v) ** 2))
        den = np.real(np.sum(np.abs(U_n.conj().T @ a_v) ** 2))
        spec[i] = num / max(den, 1e-15)
    mx = np.max(spec)
    if mx > 0: spec /= mx
    return 10 * np.log10(spec + 1e-10)


def find_peaks(spec_db, K, thr=-12):
    peaks = []
    for i in range(2, N_ANGLES - 2):
        if spec_db[i] > spec_db[i-1] and spec_db[i] > spec_db[i+1] and spec_db[i] > thr:
            peaks.append((spec_db[i], scan_deg[i]))
    peaks.sort(reverse=True)
    doas = sorted([p[1] for p in peaks[:K]])
    filt = []
    for d in doas:
        if not filt or abs(d - filt[-1]) > 3:
            filt.append(d)
    return filt


def bar(spec_db, doas, label):
    idx = np.linspace(0, N_ANGLES - 1, W).astype(int)
    vals = spec_db[idx]
    norm = np.clip((vals + 30) / 30, 0, 1)
    levels = (norm * 6).astype(int)
    chars = [' ', '.', ':', '|', '#', 'X', 'X']
    b = [chars[min(l, 6)] for l in levels]
    for doa in doas:
        pos = int((doa + 90) / 180 * (W - 1))
        pos = max(0, min(W - 1, pos))
        b[pos] = 'V'
    return f" {label:>5s} |{''.join(b)}|"


def rmse(est, true):
    if not est:
        return None
    errs = [min(abs(e - t) for t in true) for e in est]
    return np.sqrt(np.mean(np.array(errs) ** 2))


# ============================================================================
# Simple GM-PHD Tracker (2D: theta, theta_dot)
# ============================================================================
class SimpleTracker:
    def __init__(self, dt=0.1):
        self.dt = dt
        self.tracks = []  # list of [theta, theta_dot, weight, label]
        self._next = 1
        self.history = []

    def update(self, doas_rad):
        """Simple track update: associate, update, birth."""
        new_tracks = []

        # Predict existing
        predicted = []
        for th, thd, w, lbl in self.tracks:
            pred_th = th + thd * self.dt
            predicted.append([pred_th, thd, w * 0.9, lbl])

        # Associate
        used_meas = set()
        for i, (pth, pthd, pw, plbl) in enumerate(predicted):
            best_j, best_d = -1, 999
            for j, doa in enumerate(doas_rad):
                if j not in used_meas:
                    d = abs(pth - doa)
                    if d < np.radians(10) and d < best_d:
                        best_j, best_d = j, d
            if best_j >= 0:
                # Kalman-like update
                innov = doas_rad[best_j] - pth
                new_th = pth + 0.7 * innov
                new_thd = pthd + 0.3 * innov / self.dt
                new_w = min(pw + 0.2, 1.5)
                new_tracks.append([new_th, new_thd, new_w, plbl])
                used_meas.add(best_j)
            else:
                # Missed
                pw *= 0.7
                if pw > 0.05:
                    new_tracks.append([pth, pthd, pw, plbl])

        # Birth
        for j, doa in enumerate(doas_rad):
            if j not in used_meas:
                new_tracks.append([doa, 0.0, 0.15, self._next])
                self._next += 1

        self.tracks = new_tracks
        self.history.append(len([t for t in self.tracks if t[2] > 0.4]))

    def get_confirmed(self):
        return [(np.degrees(th), np.degrees(thd), w, lbl)
                for th, thd, w, lbl in self.tracks if w > 0.4]


# ============================================================================
# Main simulation
# ============================================================================
np.random.seed(42)
T = 256

scenarios = [
    ([-30, 30],             "2 sources, well separated"),
    ([-15, 15],             "2 sources, 30deg apart"),
    ([-5, 5],               "2 sources, 10deg apart (close!)"),
    ([-40, 0, 40],          "3 sources, 40deg spacing"),
    ([-30, -10, 10, 30],    "4 sources! (K=4 > M-1=3, UNDERDETERMINED)"),
    ([-50, -20, 10, 35, 60],"5 sources! (K=5, UNDERDETERMINED)"),
    ([-60, -30, 0, 30, 60], "5 sources, uniform 30deg"),
    ([-20, 20],             "Moving: converging (was -30,+30)"),
    ([0, 0],                "CROSSING at 0 deg"),
    ([20, -20],             "Post-crossing (swapped)"),
]

tracker = SimpleTracker(dt=0.1)

print()
print(f"  {'='*78}")
print(f"  DOA: CBF/MVDR/MUSIC vs COP-CBF/COP-MVDR/COP-Subspace + GM-PHD")
print(f"  M={M} | rho=2 | M_v={M_V} | T={T} | SNR=20dB | BPSK (non-Gaussian)")
print(f"  Physical array (M={M}): CBF, MVDR, MUSIC (K<={M-1})")
print(f"  Virtual  array (M_v={M_V}): COP-CBF*, COP-MVDR*, COP-Sub (K<={2*(M-1)})")
print(f"  * = K unknown OK (no source count needed)")
print(f"  {'='*78}")

all_results = []

for si, (doas, desc) in enumerate(scenarios):
    doas_arr = np.array(doas, dtype=float)
    K = len(doas_arr)
    X = gen_signal(M, T, doas_arr, snr=20)

    t0 = time.perf_counter()

    # Physical array methods
    s_cbf = cbf_spec(X)
    s_mvdr = mvdr_spec(X)
    s_music = music_spec(X, K)

    # Cumulant (shared)
    C = compute_cumulant(X)

    # COP virtual array methods
    s_cop_cbf = cop_cbf_spec(C)       # No K needed
    s_cop_mvdr = cop_mvdr_spec(C)     # No K needed
    s_cop_sub = cop_spec(C, K)        # K needed

    ms = (time.perf_counter() - t0) * 1000

    d_cbf = find_peaks(s_cbf, K)
    d_mvdr = find_peaks(s_mvdr, K)
    d_music = find_peaks(s_music, min(K, M-1))
    d_cop_cbf = find_peaks(s_cop_cbf, K)
    d_cop_mvdr = find_peaks(s_cop_mvdr, K)
    d_cop_sub = find_peaks(s_cop_sub, K)

    # Tracker uses COP-MVDR (best K-free method)
    tracker.update(np.radians(d_cop_mvdr))
    trk = tracker.get_confirmed()

    r_cbf = rmse(d_cbf, doas_arr)
    r_mvdr = rmse(d_mvdr, doas_arr)
    r_music = rmse(d_music, doas_arr)
    r_cop_cbf = rmse(d_cop_cbf, doas_arr)
    r_cop_mvdr = rmse(d_cop_mvdr, doas_arr)
    r_cop_sub = rmse(d_cop_sub, doas_arr)

    all_results.append((K, r_cbf, r_mvdr, r_music, r_cop_cbf, r_cop_mvdr, r_cop_sub, desc))

    tag = " << UNDERDETERMINED" if K > M - 1 else ""
    print(f"\n  {'-'*78}")
    print(f"  Scan {si+1:2d} | {desc}{tag}")
    print(f"  {'-'*78}")
    print(f"   deg  |-90       -60        -30       0          +30       +60        +90|")
    print(f"  --- Physical Array (M={M}) ---")
    print(bar(s_cbf, d_cbf, 'CBF'))
    print(bar(s_mvdr, d_mvdr, 'MVDR'))
    print(bar(s_music, d_music, 'MUSIC'))
    print(f"  --- COP Virtual Array (M_v={M_V}, noise-free) ---")
    print(bar(s_cop_cbf, d_cop_cbf, 'C-CBF'))
    print(bar(s_cop_mvdr, d_cop_mvdr, 'C-MVR'))
    print(bar(s_cop_sub, d_cop_sub, 'C-Sub'))
    print(f"  {'-'*78}")

    def fmt(d):
        return ', '.join(f'{x:+.0f}' for x in d) if len(d) > 0 else '---'
    def fmt_r(r):
        return f'{r:.1f}' if r is not None else 'N/A'

    print(f"  True    : {fmt(doas_arr):25s}  (K={K})")
    print(f"  CBF     : {fmt(d_cbf):25s}  RMSE={fmt_r(r_cbf):>5s}  (M={M})")
    print(f"  MVDR    : {fmt(d_mvdr):25s}  RMSE={fmt_r(r_mvdr):>5s}  (M={M})")
    print(f"  MUSIC   : {fmt(d_music):25s}  RMSE={fmt_r(r_music):>5s}  (M={M}, K needed)")
    print(f"  COP-CBF : {fmt(d_cop_cbf):25s}  RMSE={fmt_r(r_cop_cbf):>5s}  (M_v={M_V}, K-free)")
    print(f"  COP-MVDR: {fmt(d_cop_mvdr):25s}  RMSE={fmt_r(r_cop_mvdr):>5s}  (M_v={M_V}, K-free)")
    print(f"  COP-Sub : {fmt(d_cop_sub):25s}  RMSE={fmt_r(r_cop_sub):>5s}  (M_v={M_V}, K needed)")

    if trk:
        print(f"  --- GM-PHD Tracks (fed by COP-MVDR) ---")
        for doa, vel, wt, lbl in trk:
            print(f"  T{lbl:02d} | {doa:+7.1f} deg | vel={vel:+.1f} d/s | w={wt:.2f}")

    print(f"  ({ms:.0f}ms)")

# Summary table
print(f"\n  {'='*78}")
print(f"  SUMMARY: 6-Method Comparison")
print(f"  {'='*78}")
print(f"  {'Scan':>4s} | {'K':>3s} | {'CBF':>5s} | {'MVDR':>5s} | {'MUSC':>5s} | {'cCBF':>5s} | {'cMVR':>5s} | {'cSub':>5s} | Best")
print(f"  {'-'*68}")

for si, (K, rc, rm, rmu, rcc, rcm, rcs, desc) in enumerate(all_results):
    vals = {'CBF': rc, 'MVDR': rm, 'MUSC': rmu, 'cCBF': rcc, 'cMVR': rcm, 'cSub': rcs}
    valid = {k: v for k, v in vals.items() if v is not None}
    best = min(valid, key=valid.get) if valid else '?'
    und = '*' if K > M - 1 else ' '

    def f(v):
        return f'{v:5.1f}' if v is not None else '  N/A'

    print(f"  {si+1:4d} | {K:>2d}{und} | {f(rc)} | {f(rm)} | {f(rmu)} | {f(rcc)} | {f(rcm)} | {f(rcs)} | {best}")

print(f"\n  * = underdetermined (K > M-1 = {M-1})")
print(f"  Physical: CBF < MVDR < MUSIC (need K)")
print(f"  COP:      COP-CBF < COP-MVDR (K-free!) < COP-Subspace (need K)")
print(f"  COP-MVDR: best K-free method, works underdetermined, noise-immune")
print()
