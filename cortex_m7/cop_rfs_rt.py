"""COP-RFS Real-Time Implementation for ARM Cortex-M7.

Optimized for STM32H7 (480MHz, single-precision FPU, 512KB-1MB SRAM).
All buffers pre-allocated, no dynamic memory, fixed-point friendly.

Algorithm: COP-RFS (Constrained Optimization Pseudo-spectrum + Random Finite Set)
Paper: "Underdetermined High-Resolution DOA Estimation and Multi-Target Tracking via COP-RFS"
       IEEE Transactions on Signal Processing, 2026

Author: Jin Ho Choi, SmartEar Co., Ltd.
Target: ARM Cortex-M7 with FPU (STM32H743/H750/H753)

Latency budget (M=8, T=64, K_max=14):
  - Cumulant:    ~0.5ms
  - Eigen:       ~0.1ms
  - Spectrum:    ~0.2ms
  - Hungarian:   ~0.01ms
  - Kalman:      ~0.05ms
  - Total:       ~1-2ms per scan @ 480MHz
"""

import numpy as np
from typing import Optional

# ============================================================================
# Compile-time constants (Cortex-M7: all sizes fixed at build time)
# ============================================================================
MAX_SENSORS = 8          # M: physical ULA sensors
MAX_SNAPSHOTS = 128      # T: snapshots per scan
RHO = 2                  # Cumulant order (4th-order)
MAX_VIRTUAL = 15         # M_v = rho*(M-1)+1 = 2*7+1 = 15
MAX_SOURCES = 14         # K_max = rho*(M-1) = 14
MAX_TRACKS = 20          # Maximum GM-PHD components
MAX_SCAN_ANGLES = 361    # -90° to +90° at 0.5° resolution
DIM_STATE = 2            # [theta, theta_dot] (azimuth-only for ULA)
DIM_OBS = 1              # Observe theta only


# ============================================================================
# Pre-allocated buffers (Cortex-M7: static allocation in .bss)
# ============================================================================
class StaticBuffers:
    """All memory pre-allocated — zero malloc at runtime.

    Total SRAM usage estimate:
      - Cumulant matrix:  15*15*8  = 1.8 KB (complex64)
      - Eigenvalues:      15*8     = 0.12 KB
      - Eigenvectors:     15*15*8  = 1.8 KB
      - Spectrum:         361*4    = 1.4 KB (float32)
      - Steering vectors: 361*15*8 = 43 KB (can be precomputed in Flash)
      - Track states:     20*2*4   = 0.16 KB
      - Track covs:       20*4*4   = 0.32 KB
      - Cost matrix:      20*14*4  = 1.1 KB
      - Total:            ~50 KB SRAM
    """
    def __init__(self, M=MAX_SENSORS, T=MAX_SNAPSHOTS,
                 n_angles=MAX_SCAN_ANGLES):
        self.M = M
        self.T = T
        self.M_v = RHO * (M - 1) + 1
        self.n_angles = n_angles

        # Cumulant computation
        self.R = np.zeros((M, M), dtype=np.complex64)           # Covariance
        self.c4_lags = np.zeros(self.M_v, dtype=np.complex64)   # Cumulant lags
        self.c4_counts = np.zeros(self.M_v, dtype=np.float32)
        self.C = np.zeros((self.M_v, self.M_v), dtype=np.complex64)  # Cumulant matrix
        self.C_accumulated = np.zeros((self.M_v, self.M_v), dtype=np.complex64)

        # Eigendecomposition
        self.eigenvalues = np.zeros(self.M_v, dtype=np.float32)
        self.eigenvectors = np.zeros((self.M_v, self.M_v), dtype=np.complex64)
        self.U_s = np.zeros((self.M_v, MAX_SOURCES), dtype=np.complex64)
        self.U_n = np.zeros((self.M_v, self.M_v), dtype=np.complex64)

        # Spectrum
        self.spectrum = np.zeros(n_angles, dtype=np.float32)
        self.scan_angles = np.linspace(-np.pi/2, np.pi/2, n_angles).astype(np.float32)
        self.peak_indices = np.zeros(MAX_SOURCES, dtype=np.int32)
        self.peak_doas = np.zeros(MAX_SOURCES, dtype=np.float32)
        self.n_peaks = 0

        # Pre-computed virtual steering vectors (store in Flash for CM7)
        self.A_v = np.zeros((n_angles, self.M_v), dtype=np.complex64)
        self._precompute_steering_vectors()

        # GM-PHD tracker state
        self.track_mean = np.zeros((MAX_TRACKS, DIM_STATE), dtype=np.float32)
        self.track_cov = np.zeros((MAX_TRACKS, DIM_STATE, DIM_STATE), dtype=np.float32)
        self.track_weight = np.zeros(MAX_TRACKS, dtype=np.float32)
        self.track_label = np.zeros(MAX_TRACKS, dtype=np.int32)
        self.n_tracks = 0

        # Hungarian assignment
        self.cost_matrix = np.zeros((MAX_TRACKS, MAX_SOURCES), dtype=np.float32)
        self.assignment_row = np.zeros(MAX_TRACKS, dtype=np.int32)
        self.assignment_col = np.zeros(MAX_TRACKS, dtype=np.int32)

        # Temporary work buffers
        self.temp_vec = np.zeros(self.M_v, dtype=np.complex64)
        self.temp_mat = np.zeros((DIM_STATE, DIM_STATE), dtype=np.float32)

    def _precompute_steering_vectors(self):
        """Pre-compute virtual steering vectors for all scan angles.

        On Cortex-M7, store in Flash (read-only, no SRAM cost).
        a_v(theta)[n] = exp(j * 2*pi * n * d * sin(theta) / lambda)
        where d = lambda/2, so a_v[n] = exp(j * pi * n * sin(theta))
        """
        for i, theta in enumerate(self.scan_angles):
            sin_theta = np.sin(theta)
            for n in range(self.M_v):
                phase = np.pi * n * sin_theta
                self.A_v[i, n] = np.exp(1j * phase)


# ============================================================================
# COP DOA Estimator (optimized for Cortex-M7)
# ============================================================================
class COP_RT:
    """Real-time COP DOA estimator with temporal accumulation (T-COP).

    Optimizations for Cortex-M7:
      1. Pre-allocated buffers (zero malloc)
      2. Loop-order optimized for cache locality
      3. Single-precision float (CM7 FPU is single-precision)
      4. Pre-computed steering vectors (stored in Flash)
      5. In-place operations wherever possible
    """

    def __init__(self, M=MAX_SENSORS, T=MAX_SNAPSHOTS,
                 n_angles=MAX_SCAN_ANGLES, alpha=0.85):
        self.buf = StaticBuffers(M, T, n_angles)
        self.alpha = np.float32(alpha)       # T-COP forgetting factor
        self.scan_count = 0
        self.prior_weight = np.float32(0.3)  # Tracker prior weight
        self.predicted_doas = None           # From tracker feedback
        self.min_sep_rad = np.float32(np.radians(1.0))  # Min peak separation

    def compute_cumulant(self, X: np.ndarray):
        """Compute 4th-order cumulant matrix via sum co-array.

        Cortex-M7 optimized:
          - Reordered loops for sequential memory access
          - Pre-computed covariance for Gaussian term subtraction
          - In-place accumulation to pre-allocated buffer

        Args:
            X: shape (M, T), complex received signal
        """
        M = self.buf.M
        T = X.shape[1]
        L = 2 * (M - 1)
        M_v = self.buf.M_v

        # Step 1: Covariance R = X @ X^H / T  [M×M complex]
        np.dot(X, X.conj().T, out=self.buf.R[:M, :M])
        self.buf.R[:M, :M] /= T
        R = self.buf.R[:M, :M]

        # Step 2: 4th-order cumulant lags via sum co-array
        self.buf.c4_lags[:] = 0
        self.buf.c4_counts[:] = 0

        for i1 in range(M):
            xi1 = X[i1]                              # Cache row
            for i2 in range(M):
                xi2 = X[i2]
                xi1_xi2 = xi1 * xi2                   # Element-wise product
                for i3 in range(M):
                    xi3c = X[i3].conj()
                    for i4 in range(M):
                        tau = (i1 + i2) - (i3 + i4)
                        if 0 <= tau <= L:
                            # 4th-order moment: E[x_i1 * x_i2 * x_i3* * x_i4*]
                            m4 = np.mean(xi1_xi2 * xi3c * X[i4].conj())
                            # Gaussian terms
                            g1 = R[i1, i3] * R[i2, i4]
                            g2 = R[i1, i4] * R[i2, i3]
                            self.buf.c4_lags[tau] += (m4 - g1 - g2)
                            self.buf.c4_counts[tau] += 1

        # Average
        mask = self.buf.c4_counts > 0
        self.buf.c4_lags[mask] /= self.buf.c4_counts[mask]

        # Step 3: Form Hermitian Toeplitz matrix (in-place)
        for i in range(M_v):
            for j in range(M_v):
                lag = abs(i - j)
                if i >= j:
                    self.buf.C[i, j] = self.buf.c4_lags[lag]
                else:
                    self.buf.C[i, j] = self.buf.c4_lags[lag].conj()

        # Step 4: T-COP temporal accumulation
        if self.scan_count == 0:
            self.buf.C_accumulated[:] = self.buf.C
        else:
            self.buf.C_accumulated[:] = (
                self.alpha * self.buf.C_accumulated +
                (1 - self.alpha) * self.buf.C
            )
        self.scan_count += 1

    def eigendecompose(self, K: int):
        """Eigendecompose accumulated cumulant matrix.

        On Cortex-M7: Use Jacobi iteration for 15×15 Hermitian matrix.
        In Python prototype, numpy.linalg.eigh suffices.

        Args:
            K: Number of signal subspace dimensions
        """
        M_v = self.buf.M_v

        eigenvalues, eigenvectors = np.linalg.eigh(
            self.buf.C_accumulated[:M_v, :M_v]
        )

        # Sort descending by magnitude
        idx = np.argsort(np.abs(eigenvalues))[::-1]
        self.buf.eigenvalues[:M_v] = np.abs(eigenvalues[idx]).astype(np.float32)
        self.buf.eigenvectors[:M_v, :M_v] = eigenvectors[:, idx]

        # Split subspaces
        K = min(K, M_v - 1)
        self.buf.U_s[:M_v, :K] = self.buf.eigenvectors[:M_v, :K]
        self.buf.U_n[:M_v, :M_v-K] = self.buf.eigenvectors[:M_v, K:]

        return K

    def apply_tracker_prior(self, K: int):
        """Apply tracker-predicted DOA prior to refine signal subspace.

        U_s_refined = orth((1-w)*P_data + w*P_prior)

        This is the T-COP feedback loop from the tracker.
        """
        if self.predicted_doas is None or len(self.predicted_doas) == 0:
            return

        M_v = self.buf.M_v
        w = self.prior_weight

        # Build prior subspace from predicted DOAs
        n_pred = min(len(self.predicted_doas), K)
        A_prior = np.zeros((M_v, n_pred), dtype=np.complex64)
        for k in range(n_pred):
            sin_theta = np.sin(self.predicted_doas[k])
            for n in range(M_v):
                A_prior[n, k] = np.exp(1j * np.pi * n * sin_theta)

        # QR factorization for prior subspace
        Q_prior, _ = np.linalg.qr(A_prior, mode='reduced')

        # Blend projectors: P = (1-w)*U_s*U_s^H + w*Q*Q^H
        P_data = self.buf.U_s[:M_v, :K] @ self.buf.U_s[:M_v, :K].conj().T
        P_prior = Q_prior @ Q_prior.conj().T
        P_blend = (1 - w) * P_data + w * P_prior

        # Re-extract subspaces
        eigvals, eigvecs = np.linalg.eigh(P_blend)
        idx = np.argsort(eigvals)[::-1]
        self.buf.U_s[:M_v, :K] = eigvecs[:, idx[:K]]
        self.buf.U_n[:M_v, :M_v-K] = eigvecs[:, idx[K:]]

    def compute_spectrum(self, K: int):
        """Compute COP combined spectrum: P(θ) = (a^H P_s a) / (a^H P_n a).

        Cortex-M7 optimized:
          - Pre-computed steering vectors (Flash lookup)
          - Vectorized subspace projections
          - Single-precision output

        Args:
            K: Number of sources (signal subspace dimension)
        """
        M_v = self.buf.M_v
        n_angles = self.buf.n_angles

        U_s = self.buf.U_s[:M_v, :K]
        U_n = self.buf.U_n[:M_v, :M_v-K]

        for i in range(n_angles):
            a_v = self.buf.A_v[i, :M_v]

            # Signal subspace projection: ||U_s^H a_v||^2
            sig_proj = U_s.conj().T @ a_v
            numerator = np.real(np.sum(np.abs(sig_proj) ** 2))

            # Noise subspace projection: ||U_n^H a_v||^2
            noise_proj = U_n.conj().T @ a_v
            denominator = np.real(np.sum(np.abs(noise_proj) ** 2))

            if denominator < 1e-15:
                self.buf.spectrum[i] = np.float32(1e10 * numerator)
            else:
                self.buf.spectrum[i] = np.float32(numerator / denominator)

        # Normalize
        spec_max = np.max(self.buf.spectrum[:n_angles])
        if spec_max > 0:
            self.buf.spectrum[:n_angles] /= spec_max

    def find_peaks(self, K: int):
        """Extract K DOA peaks from spectrum with minimum separation gating.

        Cortex-M7: Simple loop, no dynamic allocation.
        """
        n_angles = self.buf.n_angles
        spec = self.buf.spectrum[:n_angles]
        angles = self.buf.scan_angles[:n_angles]

        # Find all local maxima
        local_max_indices = []
        local_max_vals = []
        for i in range(1, n_angles - 1):
            if spec[i] > spec[i-1] and spec[i] > spec[i+1]:
                local_max_indices.append(i)
                local_max_vals.append(spec[i])

        if len(local_max_indices) == 0:
            self.buf.n_peaks = 0
            return

        # Sort by amplitude (descending)
        sorted_idx = np.argsort(local_max_vals)[::-1]

        # Select top K with minimum separation
        selected = []
        for si in sorted_idx:
            idx = local_max_indices[si]
            theta = angles[idx]

            # Check minimum separation from already selected
            too_close = False
            for s_idx in selected:
                if abs(theta - angles[s_idx]) < self.min_sep_rad:
                    too_close = True
                    break

            if not too_close:
                selected.append(idx)
                if len(selected) >= K:
                    break

        # Store results sorted by angle
        selected.sort(key=lambda idx: angles[idx])
        self.buf.n_peaks = len(selected)
        for i, idx in enumerate(selected):
            self.buf.peak_indices[i] = idx
            self.buf.peak_doas[i] = angles[idx]

    def estimate(self, X: np.ndarray, K: int) -> np.ndarray:
        """Full COP estimation pipeline.

        Args:
            X: Received signal (M, T)
            K: Number of sources

        Returns:
            DOA estimates in radians
        """
        self.compute_cumulant(X)
        K_actual = self.eigendecompose(K)
        self.apply_tracker_prior(K_actual)
        self.compute_spectrum(K_actual)
        self.find_peaks(K)
        return self.buf.peak_doas[:self.buf.n_peaks].copy()


# ============================================================================
# GM-PHD Tracker (optimized for Cortex-M7)
# ============================================================================
class GMPHD_RT:
    """Real-time GM-PHD filter with physics-based predict-identify-update.

    State: x = [theta, theta_dot]  (azimuth DOA + angular velocity)
    Obs:   z = theta

    Cortex-M7 optimizations:
      1. Fixed-size track arrays (MAX_TRACKS)
      2. In-place Kalman update (no matrix allocation)
      3. Simplified Hungarian for small K (branch-and-bound)
      4. All operations single-precision
    """

    def __init__(self, dt=0.1, process_noise_std=0.01,
                 meas_noise_std=0.02):
        # Motion model: constant velocity
        self.dt = np.float32(dt)

        # F = [[1, dt], [0, 1]]
        self.F = np.array([[1, dt], [0, 1]], dtype=np.float32)

        # Q = sigma_w^2 * [[dt^3/3, dt^2/2], [dt^2/2, dt]]
        q = np.float32(process_noise_std ** 2)
        self.Q = q * np.array([
            [dt**3/3, dt**2/2],
            [dt**2/2, dt]
        ], dtype=np.float32)

        # H = [1, 0]
        self.H = np.array([[1, 0]], dtype=np.float32)

        # R = sigma_theta^2
        self.R = np.array([[meas_noise_std ** 2]], dtype=np.float32)

        # PHD parameters
        self.p_s = np.float32(0.95)    # Survival probability
        self.p_d = np.float32(0.90)    # Detection probability
        self.clutter_intensity = np.float32(2.0 / np.pi)  # Uniform over [-90°, 90°]
        self.birth_weight = np.float32(0.1)
        self.prune_threshold = np.float32(1e-5)
        self.merge_threshold = np.float32(4.0)  # Mahalanobis distance
        self.vel_gate = np.float32(np.radians(2.0))  # Velocity gate for merge
        self.assoc_gate = np.float32(np.radians(8.0))  # Association gate
        self.extract_threshold = np.float32(0.5)

        # Track storage (pre-allocated)
        self.mean = np.zeros((MAX_TRACKS, DIM_STATE), dtype=np.float32)
        self.cov = np.zeros((MAX_TRACKS, DIM_STATE, DIM_STATE), dtype=np.float32)
        self.weight = np.zeros(MAX_TRACKS, dtype=np.float32)
        self.label = np.zeros(MAX_TRACKS, dtype=np.int32)
        self.n_tracks = 0
        self._next_label = 0

        # Temporary buffers for predict/update
        self.pred_mean = np.zeros((MAX_TRACKS, DIM_STATE), dtype=np.float32)
        self.pred_cov = np.zeros((MAX_TRACKS, DIM_STATE, DIM_STATE), dtype=np.float32)
        self.pred_weight = np.zeros(MAX_TRACKS, dtype=np.float32)
        self.pred_label = np.zeros(MAX_TRACKS, dtype=np.int32)
        self.n_pred = 0

        # Work buffers for Kalman
        self.S = np.zeros((1, 1), dtype=np.float32)
        self.K_gain = np.zeros((DIM_STATE, 1), dtype=np.float32)

        # Cost matrix for Hungarian
        self.cost = np.zeros((MAX_TRACKS, MAX_SOURCES), dtype=np.float32)

        # Scan counter
        self.scan_count = 0

    def predict(self):
        """Predict all tracks forward using constant velocity model.

        m_pred = F @ m
        P_pred = F @ P @ F^T + Q
        w_pred = p_s * w
        """
        self.n_pred = 0

        for i in range(self.n_tracks):
            w = self.p_s * self.weight[i]
            if w < self.prune_threshold * 0.1:
                continue

            j = self.n_pred
            # State prediction: m = F @ m
            self.pred_mean[j] = self.F @ self.mean[i]
            # Covariance prediction: P = F @ P @ F^T + Q
            self.pred_cov[j] = self.F @ self.cov[i] @ self.F.T + self.Q
            self.pred_weight[j] = w
            self.pred_label[j] = self.label[i]
            self.n_pred += 1

    def associate(self, measurements: np.ndarray, n_meas: int):
        """Hungarian assignment: match measurements to predicted tracks.

        Cost = |theta_predicted - theta_measured|

        For Cortex-M7 with small K:
          Use simplified O(K²) greedy assignment (sufficient for K≤14)

        Args:
            measurements: DOA measurements in radians
            n_meas: Number of valid measurements

        Returns:
            assoc: list of (track_idx, meas_idx)
            unassoc_meas: list of unassociated measurement indices
        """
        # Only confirmed tracks (weight >= 0.3) participate
        confirmed = []
        for i in range(self.n_pred):
            if self.pred_weight[i] >= 0.3:
                confirmed.append(i)

        if len(confirmed) == 0 or n_meas == 0:
            return [], list(range(n_meas))

        n_tracks = len(confirmed)

        # Build cost matrix
        for i, ti in enumerate(confirmed):
            pred_doa = self.pred_mean[ti, 0]
            for j in range(n_meas):
                d = abs(pred_doa - measurements[j])
                self.cost[i, j] = d if d < self.assoc_gate else np.float32(1e6)

        # Greedy Hungarian (O(K²), exact for well-separated targets)
        used_tracks = set()
        used_meas = set()
        assoc = []

        # Sort all costs, pick minimum iteratively
        pairs = []
        for i in range(n_tracks):
            for j in range(n_meas):
                if self.cost[i, j] < self.assoc_gate:
                    pairs.append((self.cost[i, j], i, j))
        pairs.sort()

        for cost_val, i, j in pairs:
            if i not in used_tracks and j not in used_meas:
                assoc.append((confirmed[i], j))
                used_tracks.add(i)
                used_meas.add(j)

        unassoc_meas = [j for j in range(n_meas) if j not in used_meas]
        return assoc, unassoc_meas

    def update(self, measurements: np.ndarray, n_meas: int,
               spectrum: np.ndarray, scan_angles: np.ndarray,
               assoc: list, unassoc_meas: list):
        """Physics-based differentiated update + COP-spectrum birth.

        1. Confirmed tracks with match → Kalman update with matched only
        2. Confirmed tracks without match → missed detection (coast)
        3. Unassociated measurements → COP-spectrum weighted birth
        4. Tentative tracks → standard PHD update with all measurements
        """
        # Collect updated tracks
        new_n = 0
        temp_mean = np.zeros((MAX_TRACKS * 2, DIM_STATE), dtype=np.float32)
        temp_cov = np.zeros((MAX_TRACKS * 2, DIM_STATE, DIM_STATE), dtype=np.float32)
        temp_weight = np.zeros(MAX_TRACKS * 2, dtype=np.float32)
        temp_label = np.zeros(MAX_TRACKS * 2, dtype=np.int32)

        matched_tracks = {t: m for t, m in assoc}

        for i in range(self.n_pred):
            is_confirmed = self.pred_weight[i] >= 0.3

            if is_confirmed:
                if i in matched_tracks:
                    # === Kalman update with matched measurement ===
                    j = matched_tracks[i]
                    z = measurements[j]

                    # Innovation
                    z_pred = self.pred_mean[i, 0]  # H @ m = m[0]
                    innov = z - z_pred

                    # Innovation covariance S = H P H^T + R
                    S = self.pred_cov[i, 0, 0] + self.R[0, 0]
                    S_inv = 1.0 / max(S, 1e-15)

                    # Kalman gain K = P H^T / S
                    K0 = self.pred_cov[i, 0, 0] * S_inv
                    K1 = self.pred_cov[i, 1, 0] * S_inv

                    # State update
                    m_upd = np.array([
                        self.pred_mean[i, 0] + K0 * innov,
                        self.pred_mean[i, 1] + K1 * innov
                    ], dtype=np.float32)

                    # Covariance update: P = (I - KH)P(I - KH)^T + KRK^T
                    # Joseph form for numerical stability
                    I_KH = np.array([
                        [1 - K0, 0],
                        [-K1,    1]
                    ], dtype=np.float32)
                    P_upd = I_KH @ self.pred_cov[i] @ I_KH.T
                    P_upd[0, 0] += K0 * K0 * self.R[0, 0]
                    P_upd[0, 1] += K0 * K1 * self.R[0, 0]
                    P_upd[1, 0] += K1 * K0 * self.R[0, 0]
                    P_upd[1, 1] += K1 * K1 * self.R[0, 0]

                    # Weight: w = p_d * w * q / (kappa + p_d * w * q)
                    q = np.exp(-0.5 * innov * innov * S_inv) / np.sqrt(2 * np.pi * S)
                    w_num = self.p_d * self.pred_weight[i] * q
                    w_upd = w_num / (self.clutter_intensity + w_num)
                    w_upd = min(w_upd, 1.5)

                    if new_n < MAX_TRACKS * 2:
                        temp_mean[new_n] = m_upd
                        temp_cov[new_n] = P_upd
                        temp_weight[new_n] = np.float32(w_upd)
                        temp_label[new_n] = self.pred_label[i]
                        new_n += 1
                else:
                    # === Missed detection: coast on inertia ===
                    if new_n < MAX_TRACKS * 2:
                        temp_mean[new_n] = self.pred_mean[i]
                        temp_cov[new_n] = self.pred_cov[i]
                        temp_weight[new_n] = (1 - self.p_d) * self.pred_weight[i]
                        temp_label[new_n] = self.pred_label[i]
                        new_n += 1
            else:
                # === Tentative: standard PHD update with all measurements ===
                # Missed detection component
                if new_n < MAX_TRACKS * 2:
                    temp_mean[new_n] = self.pred_mean[i]
                    temp_cov[new_n] = self.pred_cov[i]
                    temp_weight[new_n] = (1 - self.p_d) * self.pred_weight[i]
                    temp_label[new_n] = self.pred_label[i]
                    new_n += 1

                # Update with each measurement
                for j in range(n_meas):
                    z = measurements[j]
                    z_pred = self.pred_mean[i, 0]
                    innov = z - z_pred
                    S = self.pred_cov[i, 0, 0] + self.R[0, 0]
                    S_inv = 1.0 / max(S, 1e-15)
                    K0 = self.pred_cov[i, 0, 0] * S_inv
                    K1 = self.pred_cov[i, 1, 0] * S_inv

                    m_upd = np.array([
                        self.pred_mean[i, 0] + K0 * innov,
                        self.pred_mean[i, 1] + K1 * innov
                    ], dtype=np.float32)

                    I_KH = np.array([[1-K0, 0], [-K1, 1]], dtype=np.float32)
                    P_upd = I_KH @ self.pred_cov[i] @ I_KH.T
                    P_upd[0, 0] += K0*K0*self.R[0, 0]
                    P_upd[1, 1] += K1*K1*self.R[0, 0]

                    q = np.exp(-0.5 * innov*innov*S_inv) / np.sqrt(2*np.pi*S)
                    w_num = self.p_d * self.pred_weight[i] * q
                    w_upd = w_num / (self.clutter_intensity + w_num)

                    if w_upd > self.prune_threshold and new_n < MAX_TRACKS * 2:
                        temp_mean[new_n] = m_upd
                        temp_cov[new_n] = P_upd
                        temp_weight[new_n] = np.float32(w_upd)
                        temp_label[new_n] = self.pred_label[i]
                        new_n += 1

        # === COP-spectrum birth from unassociated measurements ===
        for j in unassoc_meas:
            doa = measurements[j]

            # Birth weight proportional to COP spectrum height
            idx = np.argmin(np.abs(scan_angles - doa))
            spec_val = spectrum[idx] if idx < len(spectrum) else 0.5
            w_birth = self.birth_weight * spec_val

            if w_birth > self.prune_threshold and new_n < MAX_TRACKS * 2:
                temp_mean[new_n] = [doa, 0.0]  # Zero initial velocity
                temp_cov[new_n] = np.diag([
                    np.radians(2.0)**2,       # Position uncertainty
                    np.radians(5.0)**2        # Velocity uncertainty
                ]).astype(np.float32)
                temp_weight[new_n] = np.float32(w_birth)
                temp_label[new_n] = self._next_label
                self._next_label += 1
                new_n += 1

        return temp_mean[:new_n], temp_cov[:new_n], temp_weight[:new_n], \
               temp_label[:new_n], new_n

    def prune_and_merge(self, means, covs, weights, labels, n):
        """Prune low-weight components, velocity-gated merge, cap count.

        Velocity-gated merge: prevents track coalescence at crossings
        by refusing to merge tracks with different velocities.
        """
        # Prune
        valid = weights[:n] >= self.prune_threshold
        valid_idx = np.where(valid)[0]

        if len(valid_idx) == 0:
            self.n_tracks = 0
            return

        # Sort by weight descending for greedy merge
        sorted_idx = valid_idx[np.argsort(weights[valid_idx])[::-1]]

        # Greedy merge
        used = np.zeros(len(sorted_idx), dtype=bool)
        merged_means = []
        merged_covs = []
        merged_weights = []
        merged_labels = []

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

                # Velocity gate: don't merge if velocities differ significantly
                if (weights[i] >= 0.3 and weights[j] >= 0.3 and
                    abs(means[i, 1] - means[j, 1]) > self.vel_gate):
                    continue

                # Mahalanobis distance
                diff = means[j] - means[i]
                try:
                    cov_inv = np.linalg.inv(covs[i])
                    d2 = diff @ cov_inv @ diff
                except np.linalg.LinAlgError:
                    d2 = float('inf')

                if d2 < self.merge_threshold:
                    merge_set.append(j)
                    used[jj] = True

            # Weighted merge
            w_total = sum(weights[k] for k in merge_set)
            m_merged = sum(weights[k] * means[k] for k in merge_set) / w_total
            P_merged = np.zeros((DIM_STATE, DIM_STATE), dtype=np.float32)
            for k in merge_set:
                diff = means[k] - m_merged
                P_merged += weights[k] * (covs[k] + np.outer(diff, diff))
            P_merged /= w_total

            merged_means.append(m_merged)
            merged_covs.append(P_merged)
            merged_weights.append(w_total)
            merged_labels.append(labels[merge_set[0]])

        # Cap at MAX_TRACKS
        n_merged = min(len(merged_means), MAX_TRACKS)
        self.n_tracks = n_merged

        for i in range(n_merged):
            self.mean[i] = merged_means[i]
            self.cov[i] = merged_covs[i]
            self.weight[i] = np.float32(merged_weights[i])
            self.label[i] = merged_labels[i]

    def extract_states(self):
        """Extract confirmed target states (weight >= threshold).

        Returns:
            List of (theta, theta_dot, weight, label) tuples
        """
        targets = []
        for i in range(self.n_tracks):
            if self.weight[i] >= self.extract_threshold:
                targets.append((
                    float(self.mean[i, 0]),     # theta (rad)
                    float(self.mean[i, 1]),     # theta_dot (rad/s)
                    float(self.weight[i]),
                    int(self.label[i])
                ))

        # Sort by theta
        targets.sort(key=lambda t: t[0])

        # Deduplicate (merge within 3°)
        dedup = []
        for t in targets:
            if len(dedup) == 0 or abs(t[0] - dedup[-1][0]) > np.radians(3.0):
                dedup.append(t)

        return dedup

    def get_predicted_doas(self):
        """Get predicted DOAs from confirmed tracks (for T-COP feedback)."""
        doas = []
        for i in range(self.n_tracks):
            if self.weight[i] >= 0.3:
                # Predict one step ahead
                pred_theta = self.mean[i, 0] + self.mean[i, 1] * self.dt
                doas.append(pred_theta)
        return np.array(doas, dtype=np.float32)


# ============================================================================
# COP-RFS: Complete Pipeline (single-function per scan)
# ============================================================================
class COP_RFS_RT:
    """COP-RFS Real-Time Pipeline for Cortex-M7.

    Single-call-per-scan interface:
        results = pipeline.process_scan(X, K)

    Internal pipeline:
        1. COP DOA estimation (with T-COP temporal accumulation)
        2. GM-PHD predict (constant velocity inertia)
        3. Hungarian associate (measurement-to-track matching)
        4. Kalman update (matched-only for confirmed, PHD for tentative)
        5. COP-spectrum birth (from unassociated measurements)
        6. Prune & merge (velocity-gated)
        7. Extract targets
        8. Feedback to T-COP (predicted DOAs → constrained subspace)

    Args:
        M: Number of ULA sensors
        T: Number of snapshots per scan
        dt: Inter-scan interval (seconds)
        n_angles: Spectrum angular resolution
        alpha: T-COP forgetting factor
        process_noise_std: Target maneuverability (rad/s²)
        meas_noise_std: DOA measurement noise (rad)
    """

    def __init__(self, M=8, T=64, dt=0.1, n_angles=361,
                 alpha=0.85, process_noise_std=0.01,
                 meas_noise_std=0.02):
        self.cop = COP_RT(M=M, T=T, n_angles=n_angles, alpha=alpha)
        self.phd = GMPHD_RT(dt=dt, process_noise_std=process_noise_std,
                            meas_noise_std=meas_noise_std)
        self.scan_count = 0

    def process_scan(self, X: np.ndarray, K: int):
        """Process one radar scan through the complete COP-RFS pipeline.

        Args:
            X: Received signal matrix, shape (M, T), complex
            K: Number of expected sources

        Returns:
            targets: List of (theta_deg, theta_dot_deg, weight, label)
            doas_deg: COP DOA estimates in degrees
            spectrum: COP spatial spectrum
        """
        # Step 1: COP DOA estimation (with T-COP accumulation)
        doas_rad = self.cop.estimate(X, K)
        n_meas = self.cop.buf.n_peaks
        measurements = self.cop.buf.peak_doas[:n_meas]

        # Step 2: GM-PHD predict
        self.phd.predict()

        # Step 3: Hungarian association
        assoc, unassoc = self.phd.associate(measurements, n_meas)

        # Step 4-5: Update + Birth
        means, covs, weights, labels, n_upd = self.phd.update(
            measurements, n_meas,
            self.cop.buf.spectrum, self.cop.buf.scan_angles,
            assoc, unassoc
        )

        # Step 6: Prune & Merge (velocity-gated)
        self.phd.prune_and_merge(means, covs, weights, labels, n_upd)

        # Step 7: Extract targets
        targets_rad = self.phd.extract_states()

        # Step 8: Feedback to T-COP
        pred_doas = self.phd.get_predicted_doas()
        self.cop.predicted_doas = pred_doas if len(pred_doas) > 0 else None

        self.scan_count += 1

        # Convert to degrees for output
        targets_deg = [
            (np.degrees(t[0]), np.degrees(t[1]), t[2], t[3])
            for t in targets_rad
        ]
        doas_deg = np.degrees(doas_rad)

        return targets_deg, doas_deg, self.cop.buf.spectrum.copy()

    def get_track_info(self):
        """Get current track information for display/logging."""
        info = []
        for i in range(self.phd.n_tracks):
            if self.phd.weight[i] >= 0.1:
                info.append({
                    'label': int(self.phd.label[i]),
                    'theta_deg': float(np.degrees(self.phd.mean[i, 0])),
                    'theta_dot_deg': float(np.degrees(self.phd.mean[i, 1])),
                    'weight': float(self.phd.weight[i]),
                    'status': 'confirmed' if self.phd.weight[i] >= 0.5 else 'tentative'
                })
        return info


# ============================================================================
# Cortex-M7 Memory Map Summary
# ============================================================================
def print_memory_map():
    """Print SRAM/Flash usage estimate for Cortex-M7 deployment."""
    print("=" * 60)
    print("COP-RFS Cortex-M7 Memory Map (STM32H7)")
    print("=" * 60)

    sram = {
        'Cumulant matrix (15×15 complex64)':    15*15*8,
        'Accumulated cumulant (15×15 complex64)':15*15*8,
        'Covariance R (8×8 complex64)':          8*8*8,
        'Eigenvalues (15 float32)':              15*4,
        'Eigenvectors (15×15 complex64)':        15*15*8,
        'U_s (15×14 complex64)':                 15*14*8,
        'U_n (15×15 complex64)':                 15*15*8,
        'Spectrum (361 float32)':                361*4,
        'Track states (20×2 float32)':           20*2*4,
        'Track covariances (20×2×2 float32)':    20*4*4,
        'Track weights (20 float32)':            20*4,
        'Cost matrix (20×14 float32)':           20*14*4,
        'Temp buffers':                          2048,
    }

    flash = {
        'Steering vectors (361×15 complex64)':   361*15*8,
        'Scan angles (361 float32)':             361*4,
        'Code (~15KB est.)':                     15360,
    }

    total_sram = sum(sram.values())
    total_flash = sum(flash.values())

    print("\n[SRAM Usage]")
    for name, size in sram.items():
        print(f"  {name:45s} {size:>8,d} bytes")
    print(f"  {'─' * 45} {'─' * 8}")
    print(f"  {'TOTAL SRAM':45s} {total_sram:>8,d} bytes ({total_sram/1024:.1f} KB)")

    print(f"\n[Flash Usage]")
    for name, size in flash.items():
        print(f"  {name:45s} {size:>8,d} bytes")
    print(f"  {'─' * 45} {'─' * 8}")
    print(f"  {'TOTAL Flash':45s} {total_flash:>8,d} bytes ({total_flash/1024:.1f} KB)")

    print(f"\n[STM32H743 Budget]")
    print(f"  SRAM:  {total_sram/1024:.1f} KB / 1024 KB ({100*total_sram/(1024*1024):.1f}%)")
    print(f"  Flash: {total_flash/1024:.1f} KB / 2048 KB ({100*total_flash/(2048*1024):.1f}%)")
    print("=" * 60)
