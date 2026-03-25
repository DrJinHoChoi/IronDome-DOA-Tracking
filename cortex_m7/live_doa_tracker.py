# -*- coding: utf-8 -*-
"""COP-RFS Live DOA Estimation + Multi-Target Tracking.

Real-time system using ThinkPad X1 Carbon 4-ch mic array.
Pipeline: Audio -> STFT -> MUSIC+COP -> GM-PHD Tracker -> Display

Usage: python live_doa_tracker.py
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import sounddevice as sd
import time
import threading
from collections import deque
from scipy.linalg import toeplitz

# ============================================================================
# Constants
# ============================================================================
M = 4                    # Physical mics
MIC_SPACING = 0.065      # 65mm
SPEED_SOUND = 343.0
FS = 48000
BLOCK_SIZE = 2048        # ~42ms
ACCUM_BLOCKS = 4         # ~170ms window
FFT_SIZE = 512
HOP = 256
FREQ_LO = 300
FREQ_HI = 2600           # Below spatial Nyquist (2638Hz)
N_FREQ_BINS = 12
RHO = 2
M_V = RHO * (M - 1) + 1  # 7
K_MAX_COP = RHO * (M - 1) # 6
K_MAX_MUSIC = M - 1        # 3
N_ANGLES = 181
SCAN_DEG = np.linspace(-90, 90, N_ANGLES).astype(np.float32)
SCAN_RAD = np.radians(SCAN_DEG)
RMS_THRESH_DB = -50
PEAK_THRESH_DB = -12
DT = BLOCK_SIZE / FS     # 0.0427s
W = 68                   # Display width
MAX_TRACKS = 20


# ============================================================================
# Device Discovery
# ============================================================================
def find_mic_device():
    """Find 4-ch mic array with independent channels.

    WASAPI on Intel SST may duplicate channels (all identical),
    making DOA impossible. Prefer MME or DirectSound which provide
    true independent channel data.
    """
    devices = sd.query_devices()
    # Priority: MME > DirectSound > WASAPI (WASAPI duplicates channels on Intel SST)
    candidates = []
    for i, d in enumerate(devices):
        if d['max_input_channels'] >= M:
            name = d['name'].lower()
            if 'microphone array' in name or 'intel' in name:
                api = sd.query_hostapis(d['hostapi'])['name']
                priority = 0
                if 'MME' in api:
                    priority = 3
                elif 'DirectSound' in api:
                    priority = 2
                elif 'WDM' in api:
                    priority = 1
                else:
                    priority = 0  # WASAPI last (channel duplication issue)
                candidates.append((priority, i, d))
    if candidates:
        candidates.sort(reverse=True)
        _, idx, dev = candidates[0]
        return idx, dev
    for i, d in enumerate(devices):
        if d['max_input_channels'] >= M:
            return i, d
    return None, None


# ============================================================================
# DOA Estimator: MUSIC + COP (T-COP)
# ============================================================================
class DOAEstimator:

    def __init__(self):
        self.C_acc = None
        self.alpha = 0.6  # T-COP forgetting factor
        self.center_freq = np.sqrt(FREQ_LO * FREQ_HI)  # geometric mean ~883Hz

        # Pre-compute steering vectors
        self.A_phys = np.zeros((N_ANGLES, M), dtype=np.complex64)
        self.A_virt = np.zeros((N_ANGLES, M_V), dtype=np.complex64)
        for i, th in enumerate(SCAN_RAD):
            tau = MIC_SPACING * np.sin(th) / SPEED_SOUND
            for m in range(M):
                self.A_phys[i, m] = np.exp(-1j * 2 * np.pi * self.center_freq * m * tau)
            for n in range(M_V):
                self.A_virt[i, n] = np.exp(-1j * 2 * np.pi * self.center_freq * n * tau)

    def extract_snapshots(self, audio):
        """STFT -> narrowband snapshot matrix X (M, N_total)."""
        T_samp = audio.shape[0]
        n_frames = max(1, (T_samp - FFT_SIZE) // HOP)
        window = np.hanning(FFT_SIZE).astype(np.float32)

        f_lo_bin = max(1, int(FREQ_LO * FFT_SIZE / FS))
        f_hi_bin = min(FFT_SIZE // 2, int(FREQ_HI * FFT_SIZE / FS))
        freq_bins = np.linspace(f_lo_bin, f_hi_bin, N_FREQ_BINS).astype(int)

        snapshots = []
        for frame in range(n_frames):
            s = frame * HOP
            e = s + FFT_SIZE
            if e > T_samp:
                break
            X_fft = np.zeros((M, FFT_SIZE // 2 + 1), dtype=np.complex64)
            for m in range(M):
                X_fft[m] = np.fft.rfft(audio[s:e, m] * window)
            for fb in freq_bins:
                snapshots.append(X_fft[:, fb])

        if len(snapshots) < 4:
            return None
        return np.column_stack(snapshots)  # (M, N_total)

    def cbf_spectrum(self, X):
        """Conventional Beamforming (Delay-and-Sum) spectrum.

        P_CBF(theta) = a^H R a
        Simplest beamformer. Broad peaks, no super-resolution.
        """
        N = X.shape[1]
        R = X @ X.conj().T / N
        spec = np.zeros(N_ANGLES, dtype=np.float32)
        for i in range(N_ANGLES):
            a = self.A_phys[i]
            spec[i] = np.real(a.conj() @ R @ a)
        mx = np.max(spec)
        if mx > 0:
            spec /= mx
        return 10 * np.log10(spec + 1e-10)

    def mvdr_spectrum(self, X):
        """MVDR (Capon) Beamforming spectrum.

        P_MVDR(theta) = 1 / (a^H R^{-1} a)
        Adaptive beamformer with narrower peaks than CBF.
        Resolution limited to M-1 sources like MUSIC.
        """
        N = X.shape[1]
        R = X @ X.conj().T / N
        # Diagonal loading for stability
        R += 1e-6 * np.eye(M, dtype=np.complex64)
        try:
            R_inv = np.linalg.inv(R)
        except np.linalg.LinAlgError:
            R_inv = np.linalg.inv(R + 1e-4 * np.eye(M, dtype=np.complex64))

        spec = np.zeros(N_ANGLES, dtype=np.float32)
        for i in range(N_ANGLES):
            a = self.A_phys[i]
            den = np.real(a.conj() @ R_inv @ a)
            spec[i] = 1.0 / max(den, 1e-15)
        mx = np.max(spec)
        if mx > 0:
            spec /= mx
        return 10 * np.log10(spec + 1e-10)

    def music_spectrum(self, X):
        """MUSIC pseudo-spectrum from covariance (4x4)."""
        N = X.shape[1]
        R = X @ X.conj().T / N

        eigvals, eigvecs = np.linalg.eigh(R)
        idx = np.argsort(np.abs(eigvals))[::-1]
        eigvals_s = np.abs(eigvals[idx])
        eigvecs = eigvecs[:, idx]

        # K estimation
        ratios = eigvals_s[:-1] / (eigvals_s[1:] + 1e-30)
        K = max(1, min(int(np.argmax(ratios) + 1), M - 1))
        U_n = eigvecs[:, K:]

        spec = np.zeros(N_ANGLES, dtype=np.float32)
        for i in range(N_ANGLES):
            a = self.A_phys[i]
            proj = U_n.conj().T @ a
            den = np.real(np.sum(np.abs(proj) ** 2))
            spec[i] = 1.0 / max(den, 1e-15)

        mx = np.max(spec)
        if mx > 0:
            spec /= mx
        return 10 * np.log10(spec + 1e-10)

    def cop_spectrum(self, X):
        """COP combined spectrum with T-COP accumulation."""
        N = X.shape[1]
        L = 2 * (M - 1)
        R = X @ X.conj().T / N

        # 4th-order cumulant via sum co-array
        c4 = np.zeros(M_V, dtype=np.complex64)
        cnt = np.zeros(M_V, dtype=np.float32)

        for i1 in range(M):
            for i2 in range(M):
                xi12 = X[i1] * X[i2]
                for i3 in range(M):
                    xi3c = X[i3].conj()
                    for i4 in range(M):
                        tau = (i1 + i2) - (i3 + i4)
                        if 0 <= tau <= L:
                            m4 = np.mean(xi12 * xi3c * X[i4].conj())
                            g1 = R[i1, i3] * R[i2, i4]
                            g2 = R[i1, i4] * R[i2, i3]
                            c4[tau] += (m4 - g1 - g2)
                            cnt[tau] += 1

        mask = cnt > 0
        c4[mask] /= cnt[mask]
        C = toeplitz(c4, c4.conj())

        # T-COP accumulation
        if self.C_acc is None or self.C_acc.shape != C.shape:
            self.C_acc = C.copy()
        else:
            self.C_acc = self.alpha * self.C_acc + (1 - self.alpha) * C

        # Eigendecompose
        eigvals, eigvecs = np.linalg.eigh(self.C_acc)
        idx = np.argsort(np.abs(eigvals))[::-1]
        eigvals_s = np.abs(eigvals[idx])
        eigvecs = eigvecs[:, idx]

        ratios = eigvals_s[:-1] / (eigvals_s[1:] + 1e-30)
        K = max(1, min(int(np.argmax(ratios) + 1), M_V - 1, K_MAX_COP))
        U_s = eigvecs[:, :K]
        U_n = eigvecs[:, K:]

        # Combined spectrum
        spec = np.zeros(N_ANGLES, dtype=np.float32)
        for i in range(N_ANGLES):
            a_v = self.A_virt[i]
            s_proj = U_s.conj().T @ a_v
            num = np.real(np.sum(np.abs(s_proj) ** 2))
            n_proj = U_n.conj().T @ a_v
            den = np.real(np.sum(np.abs(n_proj) ** 2))
            spec[i] = num / max(den, 1e-15)

        mx = np.max(spec)
        if mx > 0:
            spec /= mx
        return 10 * np.log10(spec + 1e-10)

    def find_peaks(self, spec_db, max_peaks, thresh=PEAK_THRESH_DB):
        peaks = []
        for i in range(2, N_ANGLES - 2):
            if (spec_db[i] > spec_db[i-1] and spec_db[i] > spec_db[i+1]
                    and spec_db[i] > thresh):
                peaks.append((spec_db[i], SCAN_DEG[i]))
        peaks.sort(reverse=True)
        doas = sorted([p[1] for p in peaks[:max_peaks * 2]])
        filt = []
        for d in doas:
            if not filt or abs(d - filt[-1]) > 5:
                filt.append(d)
        return filt[:max_peaks]

    def estimate(self, audio):
        """Returns (cbf_db, mvdr_db, music_db, cop_db, and their DOA lists)."""
        X = self.extract_snapshots(audio)
        if X is None:
            empty = np.full(N_ANGLES, -30.0, dtype=np.float32)
            return empty, empty, empty, empty, [], [], [], []

        cbf_db = self.cbf_spectrum(X)
        mvdr_db = self.mvdr_spectrum(X)
        music_db = self.music_spectrum(X)
        cop_db = self.cop_spectrum(X)

        cbf_doas = self.find_peaks(cbf_db, K_MAX_MUSIC)
        mvdr_doas = self.find_peaks(mvdr_db, K_MAX_MUSIC)
        music_doas = self.find_peaks(music_db, K_MAX_MUSIC)
        cop_doas = self.find_peaks(cop_db, K_MAX_COP)

        return cbf_db, mvdr_db, music_db, cop_db, cbf_doas, mvdr_doas, music_doas, cop_doas


# ============================================================================
# GM-PHD Tracker: 2D state [theta, theta_dot]
# ============================================================================
class GMPHDTracker:

    def __init__(self):
        self.F = np.array([[1, DT], [0, 1]], dtype=np.float32)
        q = np.float32(0.05 ** 2)
        self.Q = q * np.array([
            [DT**3/3, DT**2/2],
            [DT**2/2, DT]
        ], dtype=np.float32)
        self.R_val = np.float32(np.radians(2.0) ** 2)

        self.p_s = 0.90
        self.p_d = 0.80
        self.clutter_int = 3.0 / np.pi
        self.birth_w = 0.15
        self.prune_thr = 1e-4
        self.merge_thr = 4.0
        self.vel_gate = np.radians(3.0)
        self.assoc_gate = np.radians(10.0)
        self.extract_thr = 0.45

        # Pre-allocated storage
        self.mean = np.zeros((MAX_TRACKS, 2), dtype=np.float32)
        self.cov = np.zeros((MAX_TRACKS, 2, 2), dtype=np.float32)
        self.weight = np.zeros(MAX_TRACKS, dtype=np.float32)
        self.label = np.zeros(MAX_TRACKS, dtype=np.int32)
        self.n = 0
        self._next_label = 1

        # Prediction buffers
        self.p_mean = np.zeros((MAX_TRACKS, 2), dtype=np.float32)
        self.p_cov = np.zeros((MAX_TRACKS, 2, 2), dtype=np.float32)
        self.p_weight = np.zeros(MAX_TRACKS, dtype=np.float32)
        self.p_label = np.zeros(MAX_TRACKS, dtype=np.int32)
        self.n_pred = 0

    def predict(self):
        self.n_pred = 0
        for i in range(self.n):
            w = self.p_s * self.weight[i]
            if w < self.prune_thr * 0.1:
                continue
            j = self.n_pred
            self.p_mean[j] = self.F @ self.mean[i]
            self.p_cov[j] = self.F @ self.cov[i] @ self.F.T + self.Q
            self.p_weight[j] = w
            self.p_label[j] = self.label[i]
            self.n_pred += 1

    def associate(self, meas_rad, n_meas):
        confirmed = [(i, self.p_mean[i, 0]) for i in range(self.n_pred)
                      if self.p_weight[i] >= 0.3]
        if not confirmed or n_meas == 0:
            return [], list(range(n_meas))

        # Greedy assignment
        pairs = []
        for ci, (ti, pred_doa) in enumerate(confirmed):
            for j in range(n_meas):
                d = abs(pred_doa - meas_rad[j])
                if d < self.assoc_gate:
                    pairs.append((d, ti, j))
        pairs.sort()

        used_t, used_m = set(), set()
        assoc = []
        for _, ti, j in pairs:
            if ti not in used_t and j not in used_m:
                assoc.append((ti, j))
                used_t.add(ti)
                used_m.add(j)

        unassoc = [j for j in range(n_meas) if j not in used_m]
        return assoc, unassoc

    def update(self, meas_rad, n_meas, cop_spec_db, assoc, unassoc):
        matched = {t: m for t, m in assoc}
        buf_m = np.zeros((MAX_TRACKS * 3, 2), dtype=np.float32)
        buf_c = np.zeros((MAX_TRACKS * 3, 2, 2), dtype=np.float32)
        buf_w = np.zeros(MAX_TRACKS * 3, dtype=np.float32)
        buf_l = np.zeros(MAX_TRACKS * 3, dtype=np.int32)
        nn = 0

        for i in range(self.n_pred):
            is_conf = self.p_weight[i] >= 0.3

            if is_conf and i in matched:
                # Kalman update with matched measurement
                j = matched[i]
                z = meas_rad[j]
                z_pred = self.p_mean[i, 0]
                innov = z - z_pred

                S = self.p_cov[i, 0, 0] + self.R_val
                S_inv = 1.0 / max(S, 1e-15)
                K0 = self.p_cov[i, 0, 0] * S_inv
                K1 = self.p_cov[i, 1, 0] * S_inv

                buf_m[nn] = [self.p_mean[i, 0] + K0 * innov,
                             self.p_mean[i, 1] + K1 * innov]
                I_KH = np.array([[1 - K0, 0], [-K1, 1]], dtype=np.float32)
                buf_c[nn] = I_KH @ self.p_cov[i] @ I_KH.T
                buf_c[nn, 0, 0] += K0 * K0 * self.R_val
                buf_c[nn, 1, 1] += K1 * K1 * self.R_val

                q = np.exp(-0.5 * innov * innov * S_inv) / np.sqrt(2 * np.pi * S)
                w_num = self.p_d * self.p_weight[i] * q
                buf_w[nn] = min(w_num / (self.clutter_int + w_num), 1.5)
                buf_l[nn] = self.p_label[i]
                nn += 1

            elif is_conf:
                # Missed detection
                if nn < MAX_TRACKS * 3:
                    buf_m[nn] = self.p_mean[i]
                    buf_c[nn] = self.p_cov[i]
                    buf_w[nn] = (1 - self.p_d) * self.p_weight[i]
                    buf_l[nn] = self.p_label[i]
                    nn += 1
            else:
                # Tentative: missed component
                if nn < MAX_TRACKS * 3:
                    buf_m[nn] = self.p_mean[i]
                    buf_c[nn] = self.p_cov[i]
                    buf_w[nn] = (1 - self.p_d) * self.p_weight[i]
                    buf_l[nn] = self.p_label[i]
                    nn += 1

                # Tentative: PHD update with all measurements
                for j in range(n_meas):
                    if nn >= MAX_TRACKS * 3:
                        break
                    z = meas_rad[j]
                    innov = z - self.p_mean[i, 0]
                    S = self.p_cov[i, 0, 0] + self.R_val
                    S_inv = 1.0 / max(S, 1e-15)
                    K0 = self.p_cov[i, 0, 0] * S_inv
                    K1 = self.p_cov[i, 1, 0] * S_inv

                    buf_m[nn] = [self.p_mean[i, 0] + K0 * innov,
                                 self.p_mean[i, 1] + K1 * innov]
                    I_KH = np.array([[1-K0, 0], [-K1, 1]], dtype=np.float32)
                    buf_c[nn] = I_KH @ self.p_cov[i] @ I_KH.T
                    buf_c[nn, 0, 0] += K0*K0*self.R_val
                    buf_c[nn, 1, 1] += K1*K1*self.R_val

                    q = np.exp(-0.5*innov*innov*S_inv) / np.sqrt(2*np.pi*S)
                    w_num = self.p_d * self.p_weight[i] * q
                    w_upd = w_num / (self.clutter_int + w_num)
                    if w_upd > self.prune_thr:
                        buf_w[nn] = w_upd
                        buf_l[nn] = self.p_label[i]
                        nn += 1

        # Birth from unassociated
        for j in unassoc:
            if nn >= MAX_TRACKS * 3:
                break
            doa = meas_rad[j]
            # COP spectrum height at this DOA
            idx = np.argmin(np.abs(SCAN_RAD - doa))
            spec_val = max(0, (cop_spec_db[idx] + 30) / 30)  # normalize 0-1
            w_birth = self.birth_w * spec_val

            if w_birth > self.prune_thr:
                buf_m[nn] = [doa, 0.0]
                buf_c[nn] = np.diag([np.radians(3.0)**2,
                                     np.radians(10.0)**2]).astype(np.float32)
                buf_w[nn] = w_birth
                buf_l[nn] = self._next_label
                self._next_label += 1
                nn += 1

        return buf_m[:nn], buf_c[:nn], buf_w[:nn], buf_l[:nn], nn

    def prune_and_merge(self, means, covs, weights, labels, n):
        # Prune
        valid = np.where(weights[:n] >= self.prune_thr)[0]
        if len(valid) == 0:
            self.n = 0
            return

        # Sort descending by weight
        sorted_idx = valid[np.argsort(weights[valid])[::-1]]
        used = np.zeros(len(sorted_idx), dtype=bool)
        merged_m, merged_c, merged_w, merged_l = [], [], [], []

        for ii in range(len(sorted_idx)):
            if used[ii]:
                continue
            i = sorted_idx[ii]
            merge_set = [i]
            used[ii] = True

            for jj in range(ii + 1, len(sorted_idx)):
                if used[jj]:
                    continue
                j = sorted_idx[jj]

                # Velocity gate
                if (weights[i] >= 0.3 and weights[j] >= 0.3 and
                        abs(means[i, 1] - means[j, 1]) > self.vel_gate):
                    continue

                diff = means[j] - means[i]
                try:
                    d2 = diff @ np.linalg.inv(covs[i]) @ diff
                except:
                    d2 = float('inf')
                if d2 < self.merge_thr:
                    merge_set.append(j)
                    used[jj] = True

            wt = sum(weights[k] for k in merge_set)
            mm = sum(weights[k] * means[k] for k in merge_set) / wt
            pp = np.zeros((2, 2), dtype=np.float32)
            for k in merge_set:
                d = means[k] - mm
                pp += weights[k] * (covs[k] + np.outer(d, d))
            pp /= wt

            merged_m.append(mm)
            merged_c.append(pp)
            merged_w.append(wt)
            merged_l.append(labels[merge_set[0]])

        n_out = min(len(merged_m), MAX_TRACKS)
        self.n = n_out
        for i in range(n_out):
            self.mean[i] = merged_m[i]
            self.cov[i] = merged_c[i]
            self.weight[i] = merged_w[i]
            self.label[i] = merged_l[i]

    def extract(self):
        """Return list of (doa_deg, vel_deg_s, weight, label, status)."""
        tracks = []
        for i in range(self.n):
            if self.weight[i] >= self.prune_thr * 10:
                status = 'confirmed' if self.weight[i] >= self.extract_thr else 'tentative'
                tracks.append((
                    np.degrees(self.mean[i, 0]),
                    np.degrees(self.mean[i, 1]),
                    float(self.weight[i]),
                    int(self.label[i]),
                    status
                ))
        # Sort by DOA, deduplicate within 5 degrees
        tracks.sort(key=lambda t: t[0])
        filt = []
        for t in tracks:
            if not filt or abs(t[0] - filt[-1][0]) > 5:
                filt.append(t)
        return filt

    def get_predicted_doas(self):
        return np.array([self.mean[i, 0] for i in range(self.n)
                         if self.weight[i] >= 0.3], dtype=np.float32)


# ============================================================================
# Live Pipeline: Audio -> DOA -> Tracking -> Display
# ============================================================================
class LiveDOATracker:

    def __init__(self):
        self.est = DOAEstimator()
        self.trk = GMPHDTracker()
        self.audio_buf = deque(maxlen=ACCUM_BLOCKS)
        self.lock = threading.Lock()

        # Shared display state (4 methods)
        empty = np.full(N_ANGLES, -30.0)
        self.cbf_db = empty.copy()
        self.mvdr_db = empty.copy()
        self.music_db = empty.copy()
        self.cop_db = empty.copy()
        self.cbf_doas = []
        self.mvdr_doas = []
        self.music_doas = []
        self.cop_doas = []
        self.tracks = []
        self.rms_db = -100.0
        self.lat_ms = 0.0
        self.scan = 0

    def process(self, audio_block):
        self.audio_buf.append(audio_block.copy())
        if len(self.audio_buf) < 2:
            return

        audio = np.vstack(list(self.audio_buf))
        rms = np.sqrt(np.mean(audio ** 2))
        rms_db = 20 * np.log10(rms + 1e-10)

        t0 = time.perf_counter()

        empty = np.full(N_ANGLES, -30.0)
        if rms_db > RMS_THRESH_DB:
            cbf_db, mvdr_db, music_db, cop_db, \
                cbf_doas, mvdr_doas, music_doas, cop_doas = self.est.estimate(audio)
        else:
            cbf_db = mvdr_db = music_db = cop_db = empty
            cbf_doas = mvdr_doas = music_doas = cop_doas = []

        # Tracker uses COP measurements
        meas_rad = np.radians(cop_doas).astype(np.float32) if cop_doas else np.array([], dtype=np.float32)
        n_meas = len(meas_rad)

        self.trk.predict()
        assoc, unassoc = self.trk.associate(meas_rad, n_meas)
        m, c, w, l, nn = self.trk.update(meas_rad, n_meas, cop_db, assoc, unassoc)
        self.trk.prune_and_merge(m, c, w, l, nn)
        tracks = self.trk.extract()

        lat_ms = (time.perf_counter() - t0) * 1000

        with self.lock:
            self.cbf_db = cbf_db
            self.mvdr_db = mvdr_db
            self.music_db = music_db
            self.cop_db = cop_db
            self.cbf_doas = list(cbf_doas)
            self.mvdr_doas = list(mvdr_doas)
            self.music_doas = list(music_doas)
            self.cop_doas = list(cop_doas)
            self.tracks = list(tracks)
            self.rms_db = rms_db
            self.lat_ms = lat_ms
            self.scan += 1


# ============================================================================
# Terminal Display (ASCII-safe)
# ============================================================================
def bar_chart(spec_db, doas, label, width=W):
    idx = np.linspace(0, N_ANGLES - 1, width).astype(int)
    vals = spec_db[idx]
    norm = np.clip((vals + 30) / 30, 0, 1)
    levels = (norm * 6).astype(int)
    chars = [' ', '.', ':', '|', '#', 'X', 'X']
    bar = [chars[min(l, 6)] for l in levels]
    for doa in doas:
        pos = int((doa + 90) / 180 * (width - 1))
        pos = max(0, min(width - 1, pos))
        bar[pos] = 'V'
    return f" {label:>5s} |{''.join(bar)}|"


def compass(doas, width=W):
    line = list('-' * width)
    c = width // 2
    line[c] = '|'
    for doa in doas:
        pos = int(c + doa * c / 90)
        pos = max(0, min(width - 1, pos))
        line[pos] = 'O'
    return f"       |{''.join(line)}|"


def display_loop(pipeline):
    while True:
        time.sleep(0.15)
        with pipeline.lock:
            cbf_db = pipeline.cbf_db.copy()
            mvdr_db = pipeline.mvdr_db.copy()
            music_db = pipeline.music_db.copy()
            cop_db = pipeline.cop_db.copy()
            cbf_doas = list(pipeline.cbf_doas)
            mvdr_doas = list(pipeline.mvdr_doas)
            music_doas = list(pipeline.music_doas)
            cop_doas = list(pipeline.cop_doas)
            tracks = list(pipeline.tracks)
            rms_db = pipeline.rms_db
            lat = pipeline.lat_ms
            scan = pipeline.scan

        if scan == 0:
            continue

        lines = []
        lines.append('')
        lines.append(f'  COP-RFS Live DOA Estimation + Multi-Target Tracking')
        lines.append(f'  4-ch Mic Array | CBF, MVDR, MUSIC (K<={K_MAX_MUSIC}) | COP (K<={K_MAX_COP}, M_v={M_V})')
        lines.append('  ' + '=' * (W + 8))

        # Axis
        ax = list(' ' * W)
        for deg, lbl in {-90: '-90', -60: '-60', -30: '-30', 0: '0',
                         30: '+30', 60: '+60', 90: '+90'}.items():
            pos = int((deg + 90) / 180 * (W - 1))
            for ci, ch in enumerate(lbl):
                if 0 <= pos + ci < W:
                    ax[pos + ci] = ch
        lines.append(f"   deg |{''.join(ax)}|")
        lines.append('  ' + '-' * (W + 8))

        # 4 spectra: CBF -> MVDR -> MUSIC -> COP (increasing resolution)
        lines.append(bar_chart(cbf_db, cbf_doas, 'CBF'))
        lines.append(bar_chart(mvdr_db, mvdr_doas, 'MVDR'))
        lines.append(bar_chart(music_db, music_doas, 'MUSIC'))
        lines.append(bar_chart(cop_db, cop_doas, 'COP'))
        lines.append(compass(cop_doas))

        lines.append('  ' + '-' * (W + 8))

        # Track table
        confirmed = [t for t in tracks if t[4] == 'confirmed']
        tentative = [t for t in tracks if t[4] == 'tentative']

        if confirmed or tentative:
            lines.append('  Tracked Targets (GM-PHD, fed by COP):')
            lines.append(f"  {'ID':>4s} | {'DOA':>7s} | {'Vel(d/s)':>8s} | {'Wt':>5s} | Status")
            lines.append('  ' + '-' * 48)
            for doa, vel, wt, lbl, status in confirmed + tentative:
                tag = '*' if status == 'confirmed' else ' '
                lines.append(f"  T{lbl:02d}{tag} | {doa:+7.1f} | {vel:+8.1f} | {wt:5.2f} | {status}")
        else:
            lines.append('  No active tracks')

        lines.append('  ' + '-' * (W + 8))

        # DOA summary for all methods
        def fmt_doas(d):
            return ', '.join(f'{x:+.0f}' for x in d) if d else '---'

        lines.append(f'  CBF:  {fmt_doas(cbf_doas):20s}  MVDR: {fmt_doas(mvdr_doas)}')
        lines.append(f'  MUSIC:{fmt_doas(music_doas):20s}  COP:  {fmt_doas(cop_doas)}')
        active = '## ACTIVE ##' if rms_db > RMS_THRESH_DB else '.. quiet ..'
        lines.append(f'  RMS:{rms_db:+.0f}dB | {lat:.0f}ms | scan {scan} | [{active}]')
        lines.append(f'  V=peak  O=COP DOA  T##*=confirmed track  Ctrl+C=stop')

        sys.stdout.write('\033[2J\033[H')
        sys.stdout.write('\n'.join(lines) + '\n')
        sys.stdout.flush()


# ============================================================================
# Main
# ============================================================================
def main():
    dev_idx, dev_info = find_mic_device()
    if dev_idx is None:
        print("ERROR: No 4-ch mic array found!")
        print(sd.query_devices())
        return

    print(f"Device: [{dev_idx}] {dev_info['name']}")
    print(f"Channels: {M} | Rate: {FS}Hz | Block: {BLOCK_SIZE} ({1000*BLOCK_SIZE/FS:.0f}ms)")
    print(f"COP: rho={RHO}, M_v={M_V}, K_max={K_MAX_COP}")
    print("Starting... (Ctrl+C to stop)\n")

    pipeline = LiveDOATracker()
    disp = threading.Thread(target=display_loop, args=(pipeline,), daemon=True)
    disp.start()

    def callback(indata, frames, time_info, status):
        if status:
            return
        audio = indata[:, :M].astype(np.float32)
        try:
            pipeline.process(audio)
        except Exception:
            pass

    try:
        with sd.InputStream(device=dev_idx, channels=M, samplerate=FS,
                            blocksize=BLOCK_SIZE, callback=callback):
            while True:
                time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n\nStopped.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
