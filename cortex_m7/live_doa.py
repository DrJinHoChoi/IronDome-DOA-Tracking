"""Live DOA Estimation from Laptop Microphone Array.

Uses ThinkPad X1 Carbon Gen 12's 4-channel Intel SST mic array
to estimate direction-of-arrival of sound sources in real-time.

Usage:
    python live_doa.py
"""

import numpy as np
import sounddevice as sd
import time
import sys
import threading
from collections import deque

# ============================================================================
# Configuration
# ============================================================================
MIC_CHANNELS = 4
MIC_SPACING_MM = 65.0
SAMPLE_RATE = 48000
BLOCK_SIZE = 4096          # ~85ms per block
ACCUMULATE_BLOCKS = 4      # Accumulate 4 blocks = ~340ms for cumulant
FREQ_BAND = (300, 3500)
SPEED_OF_SOUND = 343.0
RHO = 2
NUM_SCAN_ANGLES = 181
DISPLAY_WIDTH = 72
RMS_THRESHOLD_DB = -55     # Minimum RMS to trigger DOA estimation
PEAK_THRESHOLD_DB = -15    # Spectrum peak threshold


def find_mic_device():
    """Find best 4-ch mic array device (prefer WASAPI)."""
    devices = sd.query_devices()
    best = None
    for i, d in enumerate(devices):
        if d['max_input_channels'] >= MIC_CHANNELS:
            name = d['name'].lower()
            if 'microphone array' in name or 'intel' in name:
                api = sd.query_hostapis(d['hostapi'])['name']
                if 'WASAPI' in api:
                    return i, d
                if best is None:
                    best = (i, d)
    if best:
        return best
    # fallback
    for i, d in enumerate(devices):
        if d['max_input_channels'] >= MIC_CHANNELS:
            return i, d
    return None, None


class LiveDOA:
    """Real-time DOA estimator using SRP-PHAT + COP."""

    def __init__(self):
        self.M = MIC_CHANNELS
        self.d = MIC_SPACING_MM / 1000.0
        self.fs = SAMPLE_RATE
        self.M_v = RHO * (self.M - 1) + 1

        self.scan_deg = np.linspace(-90, 90, NUM_SCAN_ANGLES).astype(np.float32)
        self.scan_rad = np.radians(self.scan_deg)

        # Audio accumulation buffer
        self.audio_buf = deque(maxlen=ACCUMULATE_BLOCKS)

        # T-COP state
        self.C_acc = None
        self.alpha = 0.6

        # Results (shared with display thread)
        self.lock = threading.Lock()
        self.srp_spec = np.zeros(NUM_SCAN_ANGLES)
        self.cop_spec = np.zeros(NUM_SCAN_ANGLES)
        self.srp_doas = []
        self.cop_doas = []
        self.rms_db = -100.0
        self.latency_ms = 0.0
        self.scan_count = 0

    def _gcc_phat(self, x1, x2, fft_size=1024):
        """Generalized Cross-Correlation with Phase Transform."""
        X1 = np.fft.rfft(x1, n=fft_size)
        X2 = np.fft.rfft(x2, n=fft_size)
        G = X1 * X2.conj()
        denom = np.abs(G) + 1e-10
        G /= denom
        gcc = np.fft.irfft(G, n=fft_size)
        return gcc

    def estimate_srp_phat(self, audio):
        """SRP-PHAT: Steered Response Power with Phase Transform.

        Most robust broadband DOA method for real audio.
        """
        T_samp, M = audio.shape
        spectrum = np.zeros(NUM_SCAN_ANGLES)

        fft_size = 1024
        hop = 512
        n_frames = max(1, (T_samp - fft_size) // hop)

        for i, theta in enumerate(self.scan_rad):
            total_power = 0.0
            sin_t = np.sin(theta)

            for m1 in range(M):
                for m2 in range(m1 + 1, M):
                    # Expected delay between mic m1 and m2
                    tau = (m2 - m1) * self.d * sin_t / SPEED_OF_SOUND
                    delay_samples = tau * self.fs

                    # Accumulate GCC-PHAT over frames
                    for frame in range(n_frames):
                        s = frame * hop
                        e = s + fft_size
                        if e > T_samp:
                            break
                        gcc = self._gcc_phat(audio[s:e, m1], audio[s:e, m2], fft_size)

                        # Interpolate GCC at expected delay
                        d_int = int(round(delay_samples)) % fft_size
                        total_power += gcc[d_int]

            spectrum[i] = total_power

        # Normalize
        s_max = np.max(np.abs(spectrum))
        if s_max > 0:
            spectrum /= s_max

        return spectrum

    def estimate_cop(self, audio):
        """COP DOA estimation with 4th-order cumulant virtual array."""
        T_samp, M = audio.shape

        # Extract narrowband snapshots at multiple frequencies
        fft_size = 512
        hop = 256
        n_frames = max(1, (T_samp - fft_size) // hop)

        center_freq = (FREQ_BAND[0] + FREQ_BAND[1]) / 2
        f_lo = max(1, int(FREQ_BAND[0] * fft_size / self.fs))
        f_hi = min(fft_size // 2, int(FREQ_BAND[1] * fft_size / self.fs))

        # Use multiple frequency bins for robustness
        n_freq_bins = min(8, f_hi - f_lo)
        freq_bins = np.linspace(f_lo, f_hi, n_freq_bins).astype(int)
        freqs_hz = freq_bins * self.fs / fft_size

        window = np.hanning(fft_size).astype(np.float32)

        # Collect snapshots across all selected frequency bins
        all_snapshots = []
        for frame in range(n_frames):
            s = frame * hop
            e = s + fft_size
            if e > T_samp:
                break
            X_fft = np.zeros((M, fft_size // 2 + 1), dtype=np.complex64)
            for m in range(M):
                X_fft[m] = np.fft.rfft(audio[s:e, m] * window)

            for fb in freq_bins:
                all_snapshots.append(X_fft[:, fb])

        if len(all_snapshots) < 8:
            return np.zeros(NUM_SCAN_ANGLES) - 30

        X = np.column_stack(all_snapshots)  # (M, N_total)
        N = X.shape[1]

        # 4th-order cumulant
        L = 2 * (M - 1)
        M_v = L + 1
        R = X @ X.conj().T / N

        c4_lags = np.zeros(M_v, dtype=np.complex64)
        c4_counts = np.zeros(M_v)

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
                            c4_lags[tau] += (m4 - g1 - g2)
                            c4_counts[tau] += 1

        mask = c4_counts > 0
        c4_lags[mask] /= c4_counts[mask]

        from scipy.linalg import toeplitz
        C = toeplitz(c4_lags, c4_lags.conj())

        # T-COP accumulation
        if self.C_acc is None or self.C_acc.shape != C.shape:
            self.C_acc = C.copy()
        else:
            self.C_acc = self.alpha * self.C_acc + (1 - self.alpha) * C

        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(self.C_acc)
        idx = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvectors = eigenvectors[:, idx]
        eig_abs = np.abs(eigenvalues[idx])

        # Estimate K
        ratios = eig_abs[:-1] / (eig_abs[1:] + 1e-30)
        K = max(1, min(int(np.argmax(ratios) + 1), M_v - 1))

        U_s = eigenvectors[:, :K]
        U_n = eigenvectors[:, K:]

        # COP spectrum using center frequency for steering
        spectrum = np.zeros(NUM_SCAN_ANGLES)
        for i, theta in enumerate(self.scan_rad):
            tau_d = self.d * np.sin(theta) / SPEED_OF_SOUND
            a_v = np.exp(-1j * 2 * np.pi * center_freq * np.arange(M_v) * tau_d).astype(np.complex64)

            sig_proj = U_s.conj().T @ a_v
            num = np.real(np.sum(np.abs(sig_proj) ** 2))
            noise_proj = U_n.conj().T @ a_v
            den = np.real(np.sum(np.abs(noise_proj) ** 2))

            spectrum[i] = num / max(den, 1e-15)

        s_max = np.max(spectrum)
        if s_max > 0:
            spectrum /= s_max

        return spectrum

    def find_peaks(self, spectrum_db, max_peaks=6):
        """Find spectral peaks above threshold."""
        peaks = []
        for i in range(2, NUM_SCAN_ANGLES - 2):
            if (spectrum_db[i] > spectrum_db[i-1] and
                spectrum_db[i] > spectrum_db[i+1] and
                spectrum_db[i] > spectrum_db[i-2] and
                spectrum_db[i] > spectrum_db[i+2] and
                spectrum_db[i] > PEAK_THRESHOLD_DB):
                peaks.append((spectrum_db[i], self.scan_deg[i]))

        peaks.sort(reverse=True)
        doas = [p[1] for p in peaks[:max_peaks]]

        # Remove duplicates within 5°
        filtered = []
        for d in sorted(doas):
            if not filtered or abs(d - filtered[-1]) > 5:
                filtered.append(d)

        return filtered

    def process(self, audio_block):
        """Process one audio block."""
        self.audio_buf.append(audio_block.copy())

        if len(self.audio_buf) < 2:
            return

        # Concatenate accumulated blocks
        audio = np.vstack(list(self.audio_buf))

        rms = np.sqrt(np.mean(audio ** 2))
        rms_db = 20 * np.log10(rms + 1e-10)

        t0 = time.perf_counter()

        # SRP-PHAT (always works, broadband)
        srp = self.estimate_srp_phat(audio)
        srp_db = 10 * np.log10(np.abs(srp) + 1e-10)

        # COP (4th-order, needs stronger signal)
        if rms_db > RMS_THRESHOLD_DB - 10:
            cop = self.estimate_cop(audio)
            cop_db = 10 * np.log10(np.abs(cop) + 1e-10)
        else:
            cop_db = np.full(NUM_SCAN_ANGLES, -30.0)

        elapsed = (time.perf_counter() - t0) * 1000

        # Find peaks
        srp_doas = self.find_peaks(srp_db, max_peaks=3) if rms_db > RMS_THRESHOLD_DB else []
        cop_doas = self.find_peaks(cop_db, max_peaks=6) if rms_db > RMS_THRESHOLD_DB else []

        with self.lock:
            self.srp_spec = srp_db
            self.cop_spec = cop_db
            self.srp_doas = srp_doas
            self.cop_doas = cop_doas
            self.rms_db = rms_db
            self.latency_ms = elapsed
            self.scan_count += 1


def render_bar(spectrum_db, doas, width, label):
    """Render spectrum as ASCII bar with peak markers."""
    idx = np.linspace(0, len(spectrum_db)-1, width).astype(int)
    vals = spectrum_db[idx]
    angles = np.linspace(-90, 90, width)

    db_min, db_max = -30, 0
    norm = np.clip((vals - db_min) / (db_max - db_min), 0, 1)
    levels = (norm * 8).astype(int)

    blocks = ' ▁▂▃▄▅▆▇█'
    bar = list(blocks[min(l, 8)] for l in levels)

    # Mark peaks
    for doa in doas:
        pos = int((doa + 90) / 180 * (width - 1))
        pos = max(0, min(width - 1, pos))
        bar[pos] = '▼'

    return f" {label:>5s} |{''.join(bar)}|"


def render_compass(doas, width):
    """Render DOA positions on a scale."""
    line = list('·' * width)
    center = width // 2
    line[center] = '|'

    for doa in doas:
        pos = int(center + doa * center / 90)
        pos = max(0, min(width - 1, pos))
        line[pos] = '●'

    return f"       |{''.join(line)}|"


def display_loop(estimator):
    """Continuously redraw the terminal display."""
    # Build axis label
    axis_marks = {-90: '-90', -60: '-60', -30: '-30', 0: '0',
                  30: '+30', 60: '+60', 90: '+90'}

    while True:
        time.sleep(0.15)

        with estimator.lock:
            srp_spec = estimator.srp_spec.copy()
            cop_spec = estimator.cop_spec.copy()
            srp_doas = list(estimator.srp_doas)
            cop_doas = list(estimator.cop_doas)
            rms_db = estimator.rms_db
            lat = estimator.latency_ms
            scan = estimator.scan_count

        if scan == 0:
            continue

        # Build display
        W = DISPLAY_WIDTH
        lines = []
        lines.append('')
        lines.append(f'{"COP-RFS Live DOA Estimator":^{W+9}}')
        lines.append(f'{"ThinkPad X1 Carbon | 4-ch Mic Array | rho=2":^{W+9}}')
        lines.append(' ' + '-' * (W + 8))

        # Axis
        ax = list(' ' * W)
        for deg, lbl in axis_marks.items():
            pos = int((deg + 90) / 180 * (W - 1))
            for ci, ch in enumerate(lbl):
                if 0 <= pos + ci < W:
                    ax[pos + ci] = ch
        lines.append(f"  deg  |{''.join(ax)}|")
        lines.append(' ' + '-' * (W + 8))

        # SRP-PHAT spectrum + markers
        lines.append(render_bar(srp_spec, srp_doas, W, 'SRP'))
        lines.append(render_compass(srp_doas, W))
        lines.append('')

        # COP spectrum + markers
        lines.append(render_bar(cop_spec, cop_doas, W, 'COP'))
        lines.append(render_compass(cop_doas, W))
        lines.append(' ' + '-' * (W + 8))

        # Info
        srp_str = ', '.join(f'{d:+.0f}°' for d in srp_doas) if srp_doas else '---'
        cop_str = ', '.join(f'{d:+.0f}°' for d in cop_doas) if cop_doas else '---'

        active = '■ ACTIVE' if rms_db > RMS_THRESHOLD_DB else '□ quiet'
        lines.append(f'  SRP-PHAT : {srp_str:30s} (max 3 sources)')
        lines.append(f'  COP-4th  : {cop_str:30s} (max 6 sources)')
        lines.append(f'  RMS: {rms_db:+.1f} dB  |  {lat:.0f}ms  |  scan {scan}  |  {active}')
        lines.append(f'  ▼ = estimated source direction    ● = DOA position')
        lines.append(f'  Speak or clap to see direction change!')

        # Clear + print
        sys.stdout.write('\033[2J\033[H')
        sys.stdout.write('\n'.join(lines) + '\n')
        sys.stdout.flush()


def main():
    dev_idx, dev_info = find_mic_device()
    if dev_idx is None:
        print("ERROR: No 4-ch mic array found!")
        print(sd.query_devices())
        return

    print(f"Using: [{dev_idx}] {dev_info['name']}")
    print(f"Channels: {MIC_CHANNELS}, Rate: {SAMPLE_RATE} Hz")
    print("Starting... (Ctrl+C to stop)\n")

    estimator = LiveDOA()

    # Start display thread
    disp_thread = threading.Thread(target=display_loop, args=(estimator,), daemon=True)
    disp_thread.start()

    def callback(indata, frames, time_info, status):
        if status:
            return
        audio = indata[:, :MIC_CHANNELS].astype(np.float32)
        try:
            estimator.process(audio)
        except Exception:
            pass

    try:
        with sd.InputStream(
            device=dev_idx,
            channels=MIC_CHANNELS,
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            callback=callback
        ):
            while True:
                time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n\nStopped.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
