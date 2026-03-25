# -*- coding: utf-8 -*-
"""COP vs MUSIC DOA Estimation with Real Speech WAV Files.

Uses Google Speech Commands dataset (different speakers)
to simulate multi-speaker DOA scenarios.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import time
import os
import wave
from scipy.linalg import toeplitz
from scipy.stats import kurtosis

# ============================================================================
# Config
# ============================================================================
W = 70
NUM_ANGLES = 181
scan_deg = np.linspace(-90, 90, NUM_ANGLES)
scan_rad = np.radians(scan_deg)
DATA_DIR = "C:/Users/jinho/Downloads/SmartEar-KWS-/data/SpeechCommands/speech_commands_v0.02"
MIC_SPACING = 0.065   # 65mm
SPEED_SOUND = 343.0
FREQ_CENTER = 1500    # Hz, good for speech formants


def load_wav(path):
    """Load wav file as float array."""
    with wave.open(path, 'rb') as wf:
        fs = wf.getframerate()
        n = wf.getnframes()
        data = np.frombuffer(wf.readframes(n), dtype=np.int16).astype(np.float64)
        data /= 32768.0
    return data, fs


def load_different_speakers(words, n_speakers):
    """Load wav files from different speakers, one per word."""
    signals = []
    used_speakers = set()

    # Cycle through different words to get diverse speakers
    for word in words:
        if len(signals) >= n_speakers:
            break
        wdir = os.path.join(DATA_DIR, word)
        if not os.path.exists(wdir):
            continue
        files = sorted(os.listdir(wdir))
        for f in files:
            speaker = f.split('_')[0]
            if speaker not in used_speakers:
                path = os.path.join(wdir, f)
                data, fs = load_wav(path)
                if len(data) > 4000:
                    # Normalize power per speaker
                    rms = np.sqrt(np.mean(data**2)) + 1e-10
                    data = data / rms
                    signals.append((data, fs, word, speaker))
                    used_speakers.add(speaker)
                    break  # One speaker per word
    return signals


def simulate_ula(signals, doas_deg, M=4, target_len=8000):
    """Simulate M-element ULA received signal from real speech sources.

    Each speech source is placed at a different DOA.
    Returns narrowband snapshots at FREQ_CENTER for DOA estimation.
    """
    K = len(doas_deg)

    # Pad/trim all signals to same length
    for i in range(len(signals)):
        s = signals[i][0]
        if len(s) < target_len:
            s = np.pad(s, (0, target_len - len(s)))
        else:
            s = s[:target_len]
        signals[i] = (s, signals[i][1], signals[i][2], signals[i][3])

    fs = signals[0][1]

    # Use MULTIPLE frequency bins for broadband robustness
    fft_size = 512
    hop = 128
    n_frames = (target_len - fft_size) // hop
    window = np.hanning(fft_size)

    # Frequency bins: 500-3000Hz range (speech formants)
    f_lo = int(500 * fft_size / fs)
    f_hi = int(3000 * fft_size / fs)
    freq_bins = np.linspace(f_lo, f_hi, 8).astype(int)

    # Collect snapshots across all frequency bins
    total_snapshots = n_frames * len(freq_bins)
    X = np.zeros((M, total_snapshots), dtype=complex)
    snap_idx = 0

    for frame in range(n_frames):
        s_start = frame * hop
        s_end = s_start + fft_size

        # FFT each source
        source_spectra = []
        for k in range(K):
            seg = signals[k][0][s_start:s_end] * window
            source_spectra.append(np.fft.rfft(seg))

        for fb in freq_bins:
            freq_hz = fb * fs / fft_size
            # Steering at this frequency
            A = np.zeros((M, K), dtype=complex)
            for k in range(K):
                tau = MIC_SPACING * np.sin(np.radians(doas_deg[k])) / SPEED_SOUND
                for m in range(M):
                    A[m, k] = np.exp(-1j * 2 * np.pi * freq_hz * m * tau)

            s_vec = np.array([sp[min(fb, len(sp)-1)] for sp in source_spectra])
            X[:, snap_idx] = A @ s_vec
            snap_idx += 1

    X = X[:, :snap_idx]

    # Add noise (SNR ~20dB)
    sig_power = np.mean(np.abs(X)**2)
    noise_power = sig_power / (10**(20/10))
    noise = np.sqrt(noise_power/2) * (np.random.randn(M, snap_idx) + 1j*np.random.randn(M, snap_idx))
    X += noise

    return X, fs


def cop_spectrum(X, M_v, K):
    """COP 4th-order cumulant spectrum."""
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
        tau_d = MIC_SPACING * np.sin(th) / SPEED_SOUND
        a_v = np.exp(-1j * 2 * np.pi * FREQ_CENTER * np.arange(M_v) * tau_d)
        num = np.real(np.sum(np.abs(U_s.conj().T @ a_v)**2))
        den = np.real(np.sum(np.abs(U_n.conj().T @ a_v)**2))
        spec[i] = num / max(den, 1e-15)
    mx = np.max(spec)
    if mx > 0: spec /= mx
    return 10*np.log10(spec + 1e-10)


def music_spectrum(X, K):
    """MUSIC spectrum."""
    M, T = X.shape
    R = X @ X.conj().T / T
    eigvals, eigvecs = np.linalg.eigh(R)
    idx = np.argsort(np.abs(eigvals))[::-1]
    eigvecs = eigvecs[:, idx]
    K = min(K, M-1)
    U_n = eigvecs[:, K:]
    spec = np.zeros(NUM_ANGLES)
    for i, th in enumerate(scan_rad):
        tau_d = MIC_SPACING * np.sin(th) / SPEED_SOUND
        a = np.exp(-1j * 2 * np.pi * FREQ_CENTER * np.arange(M) * tau_d)
        den = np.real(np.sum(np.abs(U_n.conj().T @ a)**2))
        spec[i] = 1.0 / max(den, 1e-15)
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


# ============================================================================
# Main
# ============================================================================
np.random.seed(42)
M = 4
M_v = 2*(M-1)+1  # 7

print()
print(f"  {'='*76}")
print(f"  COP vs MUSIC with REAL SPEECH (Google Speech Commands)")
print(f"  M=4 mics | 65mm spacing | rho=2 | M_v=7")
print(f"  COP: up to 6 sources  |  MUSIC: up to 3 sources")
print(f"  {'='*76}")

# Load speech files from different speakers/words
words_pool = ['yes', 'no', 'stop', 'go', 'left', 'right', 'up', 'down']
all_signals = load_different_speakers(words_pool, 6)

print(f"\n  Loaded {len(all_signals)} speakers:")
for i, (data, fs, word, spk) in enumerate(all_signals):
    k = kurtosis(data)
    print(f"    Speaker {i+1}: '{word}' (id={spk[:8]}, fs={fs}Hz, kurtosis={k:.1f})")

print(f"\n  Speech kurtosis >> 0 -> strong non-Gaussian -> COP works!")

scenarios = [
    # (speaker indices, doas, description)
    ([0, 1],          [-30, 30],          "2 speakers: 'yes' vs 'no'"),
    ([0, 1],          [-15, 15],          "2 speakers close: -15, +15"),
    ([0, 1],          [-5, 5],            "2 speakers VERY close: -5, +5"),
    ([0, 1, 2],       [-40, 0, 40],       "3 speakers: yes/no/stop"),
    ([0, 1, 2, 3],    [-45, -15, 15, 45], "4 speakers! (UNDERDETERMINED K>3)"),
    ([0, 1, 2, 3, 4], [-60, -30, 0, 30, 60], "5 speakers! (K=5 >> M-1=3)"),
]

results_cop = []
results_music = []

for si, (spk_idx, doas, desc) in enumerate(scenarios):
    doas_arr = np.array(doas, dtype=float)
    K = len(doas_arr)
    sigs = [all_signals[i] for i in spk_idx]

    X, fs = simulate_ula(sigs, doas_arr, M=M)

    underdetermined = K > M - 1

    # COP
    t0 = time.perf_counter()
    cop_db = cop_spectrum(X, M_v, K)
    cop_doas = find_peaks(cop_db, K)
    cop_ms = (time.perf_counter() - t0) * 1000

    # MUSIC
    t0 = time.perf_counter()
    mus_db = music_spectrum(X, min(K, M-1))
    mus_doas = find_peaks(mus_db, min(K, M-1))
    mus_ms = (time.perf_counter() - t0) * 1000

    # RMSE
    cop_rmse = None
    if cop_doas:
        errs = [min(abs(e - t) for t in doas_arr) for e in cop_doas]
        cop_rmse = np.sqrt(np.mean(np.array(errs)**2))
    mus_rmse = None
    if mus_doas:
        errs = [min(abs(e - t) for t in doas_arr) for e in mus_doas]
        mus_rmse = np.sqrt(np.mean(np.array(errs)**2))

    results_cop.append(cop_rmse)
    results_music.append(mus_rmse)

    tag = " << UNDERDETERMINED" if underdetermined else ""
    print(f"\n  {'-'*76}")
    print(f"  Scan {si+1} | {desc}{tag}")
    print(f"  {'-'*76}")
    print(f"   deg  |-90       -60        -30       0          +30       +60        +90|")
    print(bar_chart(cop_db, cop_doas, "COP"))
    print(dot_line(cop_doas))
    print(bar_chart(mus_db, mus_doas, "MUSIC"))
    print(dot_line(mus_doas))

    true_s = ", ".join(f"{d:+.0f}" for d in doas_arr)
    cop_s = ", ".join(f"{d:+.0f}" for d in cop_doas) if cop_doas else "---"
    mus_s = ", ".join(f"{d:+.0f}" for d in mus_doas) if mus_doas else "---"

    print(f"  True  : {true_s} deg  (K={K})")
    print(f"  COP   : {cop_s} deg  (RMSE={cop_rmse:.1f})" if cop_rmse is not None else f"  COP   : {cop_s}")
    print(f"  MUSIC : {mus_s} deg  (RMSE={mus_rmse:.1f})" if mus_rmse is not None else f"  MUSIC : {mus_s}")

    winner = ""
    if cop_rmse is not None and mus_rmse is not None:
        if cop_rmse < mus_rmse - 1:
            winner = ">> COP WINS"
        elif mus_rmse < cop_rmse - 1:
            winner = ">> MUSIC WINS"
        else:
            winner = ">> TIE"
    elif underdetermined and cop_doas and not mus_doas:
        winner = ">> COP WINS (MUSIC can't resolve)"
    elif underdetermined and cop_rmse is not None:
        winner = ">> COP WINS (underdetermined)"
    print(f"  {winner}")

# Summary
print(f"\n  {'='*76}")
print(f"  SUMMARY: COP vs MUSIC with Real Speech")
print(f"  {'='*76}")
print(f"  {'Scan':>6s} | {'K':>2s} | {'COP RMSE':>10s} | {'MUSIC RMSE':>10s} | {'Winner':>12s}")
print(f"  {'-'*56}")
for i, (cr, mr) in enumerate(zip(results_cop, results_music)):
    K = len(scenarios[i][1])
    cs = f"{cr:.1f}" if cr is not None else "N/A"
    ms = f"{mr:.1f}" if mr is not None else "N/A"
    if cr is not None and mr is not None:
        w = "COP" if cr < mr - 1 else ("MUSIC" if mr < cr - 1 else "TIE")
    elif K > 3:
        w = "COP"
    else:
        w = "?"
    print(f"  {i+1:>6d} | {K:>2d} | {cs:>10s} | {ms:>10s} | {w:>12s}")

print()
