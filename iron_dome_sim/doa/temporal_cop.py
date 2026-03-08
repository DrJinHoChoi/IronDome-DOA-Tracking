"""Temporal COP (T-COP): Tracker-Aided Temporal Cumulant Accumulation.

Novel extension of the 2rho-th order Subspace Constrained Optimization
that exploits temporal consistency from multi-target tracking to improve
cumulant estimation and DOA accuracy.

Key innovations:
1. Temporal cumulant accumulation with exponential forgetting
2. Tracker-predicted DOA priors constrain the signal subspace
3. Adaptive subspace dimension based on track state
4. Scan-to-scan cumulant coherent integration

This creates a feedback loop:
    COP DOA estimation → Tracker update → Predicted DOAs →
    → Constrained COP (next scan) → ...

The temporal integration effectively increases the snapshot count
without requiring longer observation windows, improving performance
for non-stationary signals.

Patent-relevant novel contributions:
- Tracker-predicted subspace constraint in higher-order COP
- Exponentially weighted temporal cumulant matrix
- Adaptive signal subspace dimension from track count

Reference (base algorithm):
    Choi & Yoo, IEEE TSP 2015.
"""

import numpy as np
from scipy.linalg import toeplitz
from .base import DOAEstimator
from .spectrum import find_peaks_doa
from ..signal_model.cumulant import compute_cumulant_matrix


class TemporalCOP(DOAEstimator):
    """Temporal COP: Tracking-aided temporal cumulant accumulation.

    Extends the standard COP by:
    1. Accumulating cumulant matrices across scans with exponential forgetting
    2. Using tracker-predicted DOAs to constrain the subspace decomposition
    3. Narrowing the spectral search around predicted DOA regions
    4. Adapting the signal subspace dimension from confirmed track count

    Args:
        array: ULA array object.
        rho: Cumulant order parameter (default: 2).
        alpha: Temporal forgetting factor (0 < alpha <= 1).
               alpha=1: no forgetting (use all history equally).
               alpha=0.8: recent scans weighted ~5x more than 5 scans ago.
        prior_weight: Weight of tracker prior in subspace constraint (0-1).
        search_width_deg: Angular search width around predicted DOAs (degrees).
        num_sources: Fixed number of sources (None = adaptive from tracker).
    """

    def __init__(self, array, rho=2, alpha=0.85, prior_weight=0.3,
                 search_width_deg=15.0, num_sources=None):
        super().__init__(array, num_sources)
        self.rho = rho
        self.alpha = alpha
        self.prior_weight = prior_weight
        self.search_width = np.radians(search_width_deg)
        self.name = f"T-COP-{2*rho}th"

        # Temporal state
        self.C_accumulated = None  # Accumulated cumulant matrix
        self.scan_count = 0
        self.predicted_doas = None  # From tracker
        self.predicted_covs = None  # DOA uncertainties from tracker
        self.n_confirmed_tracks = 0

    @property
    def is_underdetermined(self):
        return True

    @property
    def max_sources(self):
        return self.array.max_sources(self.rho)

    def set_tracker_predictions(self, predicted_doas, predicted_covs=None,
                                n_confirmed=0):
        """Receive predictions from the tracker (feedback loop).

        Called before each scan's DOA estimation.

        Args:
            predicted_doas: Predicted DOA angles from tracker (radians).
            predicted_covs: DOA variance for each prediction.
            n_confirmed: Number of confirmed tracks.
        """
        self.predicted_doas = np.array(predicted_doas) if len(predicted_doas) > 0 else None
        self.predicted_covs = np.array(predicted_covs) if predicted_covs is not None else None
        self.n_confirmed_tracks = n_confirmed

    def estimate(self, X, scan_angles=None):
        """Estimate DOA with temporal accumulation and tracker priors.

        Args:
            X: Received signal matrix, shape (M, T).
            scan_angles: Angular scan grid in radians.

        Returns:
            doa_estimates: Estimated DOA angles in radians.
            spectrum: Spatial spectrum values.
        """
        if scan_angles is None:
            scan_angles = self._default_scan_angles()

        # Step 1: Compute current scan's cumulant
        C_current = compute_cumulant_matrix(X, self.rho)
        M_v = C_current.shape[0]

        # Step 2: Temporal cumulant accumulation
        C_temporal = self._accumulate_cumulant(C_current)

        # Step 3: Tracker-aided subspace constraint
        K = self._determine_num_sources(C_temporal)

        # Step 4: Compute spectrum with prior-constrained subspace
        P = self._compute_constrained_spectrum(C_temporal, scan_angles, K)

        # Step 5: Peak detection (prioritize regions near predicted DOAs)
        doa_estimates = self._prior_guided_peak_detection(P, scan_angles, K)

        self.scan_count += 1
        return doa_estimates, P

    def spectrum(self, X, scan_angles):
        """Compute temporal COP spectrum."""
        C_current = compute_cumulant_matrix(X, self.rho)
        C_temporal = self._accumulate_cumulant(C_current)
        K = self._determine_num_sources(C_temporal)
        return self._compute_constrained_spectrum(C_temporal, scan_angles, K)

    def _accumulate_cumulant(self, C_current):
        """Exponentially weighted temporal cumulant accumulation.

        C_acc(n) = alpha * C_acc(n-1) + (1 - alpha) * C(n)

        This effectively increases the equivalent snapshot count by
        combining information from multiple scans while adapting to
        signal non-stationarity through the forgetting factor.
        """
        if self.C_accumulated is None:
            self.C_accumulated = C_current.copy()
        else:
            # Ensure dimension match
            if self.C_accumulated.shape == C_current.shape:
                self.C_accumulated = (self.alpha * self.C_accumulated +
                                      (1 - self.alpha) * C_current)
            else:
                self.C_accumulated = C_current.copy()

        return self.C_accumulated.copy()

    def _determine_num_sources(self, C):
        """Determine signal subspace dimension.

        Uses tracker information when available:
        - If confirmed tracks exist, use track count as lower bound
        - Eigenvalue analysis for upper bound
        - Weighted combination for final estimate
        """
        if self.num_sources is not None:
            return self.num_sources

        # Eigenvalue-based estimation
        eigenvalues = np.sort(np.abs(np.linalg.eigvalsh(C)))[::-1]
        M_v = len(eigenvalues)

        # Ratio test
        ratios = eigenvalues[:-1] / (eigenvalues[1:] + 1e-30)
        K_eigen = np.argmax(ratios) + 1
        K_eigen = max(1, min(K_eigen, M_v - 1))

        # Tracker-based count
        K_tracker = self.n_confirmed_tracks

        if K_tracker > 0:
            # Use max of tracker count and eigenvalue estimate
            # Tracker provides a reliable lower bound
            K = max(K_tracker, K_eigen)
        else:
            K = K_eigen

        return min(K, M_v - 1)

    def _compute_constrained_spectrum(self, C, scan_angles, K):
        """Compute COP spectrum with tracker-aided subspace constraint.

        When tracker predictions are available:
        1. Project cumulant matrix using prior-informed subspace
        2. Enhance signal subspace along predicted DOA directions
        3. Compute combined spectrum with sharpened peaks at predicted locations
        """
        M_v = C.shape[0]

        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(C)
        idx = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        K = min(K, M_v - 1)
        U_s = eigenvectors[:, :K]
        U_n = eigenvectors[:, K:]

        # Apply tracker prior to refine subspace (novel contribution)
        if self.predicted_doas is not None and self.prior_weight > 0:
            U_s, U_n = self._apply_prior_constraint(
                U_s, U_n, M_v, K
            )

        # Compute spectrum
        P = np.zeros(len(scan_angles))
        for i, theta in enumerate(scan_angles):
            a_v = self.array.virtual_steering_vector(theta, self.rho)
            if len(a_v) != M_v:
                a_v = a_v[:M_v] if len(a_v) > M_v else np.pad(a_v, (0, M_v - len(a_v)))

            # Combined signal × noise spectrum
            sig_proj = U_s.conj().T @ a_v
            numerator = np.real(np.sum(np.abs(sig_proj) ** 2))

            noise_proj = U_n.conj().T @ a_v
            denominator = np.real(np.sum(np.abs(noise_proj) ** 2))

            if denominator < 1e-15:
                P[i] = 1e10 * (numerator + 1e-15)
            else:
                P[i] = numerator / denominator

        # Normalize
        P_max = np.max(P)
        if P_max > 0:
            P /= P_max

        return P

    def _apply_prior_constraint(self, U_s, U_n, M_v, K):
        """Apply tracker-predicted DOA prior to refine subspace estimate.

        Novel contribution: Uses predicted DOAs to construct a prior
        signal subspace, then blends it with the data-driven subspace.

        This improves robustness when:
        - SNR is low (data subspace is noisy)
        - Sources are closely spaced (data subspace is ambiguous)
        - Number of snapshots is limited

        The blending formula:
            U_s_refined = orth(w * U_s_prior + (1-w) * U_s_data)
        """
        w = self.prior_weight

        # Construct prior signal subspace from predicted DOAs
        prior_vectors = []
        for doa in self.predicted_doas:
            a_v = self.array.virtual_steering_vector(doa, self.rho)
            if len(a_v) != M_v:
                a_v = a_v[:M_v] if len(a_v) > M_v else np.pad(a_v, (0, M_v - len(a_v)))
            prior_vectors.append(a_v)

        if len(prior_vectors) == 0:
            return U_s, U_n

        A_prior = np.column_stack(prior_vectors)

        # Prior signal subspace (via QR)
        Q_prior, _ = np.linalg.qr(A_prior, mode='reduced')
        K_prior = Q_prior.shape[1]

        # Blend data-driven and prior subspaces
        K_blend = min(K, M_v - 1)
        K_use = min(K_blend, U_s.shape[1])

        # Weighted combination of projectors
        P_data = U_s[:, :K_use] @ U_s[:, :K_use].conj().T
        P_prior = Q_prior @ Q_prior.conj().T

        P_blended = (1 - w) * P_data + w * P_prior

        # Re-extract subspaces from blended projector
        eigvals_b, eigvecs_b = np.linalg.eigh(P_blended)
        idx_b = np.argsort(eigvals_b)[::-1]
        eigvecs_b = eigvecs_b[:, idx_b]

        U_s_refined = eigvecs_b[:, :K_blend]
        U_n_refined = eigvecs_b[:, K_blend:]

        return U_s_refined, U_n_refined

    def _prior_guided_peak_detection(self, spectrum, scan_angles, K):
        """Peak detection guided by tracker predictions.

        When no predictions available: standard peak detection (same as COP).
        When predictions available:
        1. Search near predicted DOAs (high confidence regions)
        2. Search remaining spectrum for new targets

        This addresses the missed detection problem at high K.
        """
        if self.predicted_doas is None or len(self.predicted_doas) == 0:
            # No tracker info: standard peak detection (identical to COP)
            return find_peaks_doa(spectrum, scan_angles, K)

        # With tracker predictions: find extra peaks, then prioritize
        all_peaks = find_peaks_doa(spectrum, scan_angles, K + 5)

        # Phase 1: Confirm peaks near predicted DOAs
        confirmed = []
        remaining_peaks = list(all_peaks)

        for pred_doa in self.predicted_doas:
            if len(remaining_peaks) == 0:
                break
            distances = np.abs(np.array(remaining_peaks) - pred_doa)
            best_idx = np.argmin(distances)

            if distances[best_idx] < self.search_width:
                confirmed.append(remaining_peaks[best_idx])
                remaining_peaks.pop(best_idx)

        # Phase 2: Add unmatched peaks (new targets) by spectrum height
        n_remaining = K - len(confirmed)
        if n_remaining > 0 and len(remaining_peaks) > 0:
            remaining_heights = [spectrum[np.argmin(np.abs(scan_angles - p))]
                                 for p in remaining_peaks]
            sorted_idx = np.argsort(remaining_heights)[::-1]
            for i in range(min(n_remaining, len(remaining_peaks))):
                confirmed.append(remaining_peaks[sorted_idx[i]])

        return np.sort(np.array(confirmed))

    def reset(self):
        """Reset temporal state."""
        self.C_accumulated = None
        self.scan_count = 0
        self.predicted_doas = None
        self.predicted_covs = None
        self.n_confirmed_tracks = 0
