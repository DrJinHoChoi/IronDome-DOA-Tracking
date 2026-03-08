"""Subspace Constrained Optimization (COP) for underdetermined DOA estimation.

Implements the 2rho-th order source-signal/noise subspace constrained
optimization algorithm from:

    Choi, J.H. and Yoo, C.D., "Underdetermined High-Resolution DOA Estimation:
    A 2rho-th-Order Source-Signal/Noise Subspace Constrained Optimization,"
    IEEE Trans. Signal Processing, vol. 63, no. 7, pp. 1858-1873, 2015.

Key properties:
    - Resolves up to 2*rho*(M-1) sources with M sensors (underdetermined)
    - Noise elimination via higher-order cumulants (Gaussian noise vanishes)
    - Combined signal-subspace x noise-subspace spectrum for sharp peaks
"""

import numpy as np
from .base import DOAEstimator
from ..signal_model.cumulant import compute_cumulant_matrix


class SubspaceCOP(DOAEstimator):
    """2rho-th Order Subspace Constrained Optimization DOA estimator.

    This is the proposed algorithm that achieves underdetermined DOA estimation
    by exploiting higher-order statistics and subspace constraints.

    Args:
        array: ULA array object.
        rho: Order parameter (default: 2 → 4th-order cumulant).
             Higher rho gives more resolution but needs more snapshots.
        num_sources: Number of sources (None = auto-detect).
        spectrum_type: "combined" (signal × noise, sharpest peaks),
                       "signal" (signal subspace only),
                       "noise" (noise subspace only, MUSIC-like).
    """

    def __init__(self, array, rho=2, num_sources=None, spectrum_type="combined"):
        super().__init__(array, num_sources)
        self.rho = rho
        self.spectrum_type = spectrum_type
        self.name = f"COP-{2*rho}th"

    @property
    def is_underdetermined(self):
        return True

    @property
    def max_sources(self):
        return self.array.max_sources(self.rho)

    def estimate(self, X, scan_angles=None):
        """Estimate DOA using subspace constrained optimization.

        Args:
            X: Received signal matrix, shape (M, T).
            scan_angles: Angular scan grid in radians.

        Returns:
            doa_estimates: Estimated DOA angles in radians.
            P: Spatial spectrum values.
        """
        if scan_angles is None:
            scan_angles = self._default_scan_angles()

        P = self.spectrum(X, scan_angles)

        # Determine number of sources
        K = self.num_sources
        if K is None:
            K = self._estimate_num_sources_cumulant(X)

        # Find K highest peaks
        from .spectrum import find_peaks_doa
        doa_estimates = find_peaks_doa(P, scan_angles, K)

        return doa_estimates, P

    def spectrum(self, X, scan_angles):
        """Compute COP spatial spectrum.

        Steps:
        1. Compute 2rho-th order cumulant matrix C_{2rho}
        2. Eigendecompose C_{2rho} → signal subspace U_s, noise subspace U_n
        3. For each look direction theta:
           a. Construct virtual steering vector a_v(theta)
           b. Solve signal-constrained COP → w_s(theta)
           c. Solve noise-constrained COP → w_n(theta)
           d. P(theta) = 1/||w_s||^2 * ||w_n||^2 (combined)

        Args:
            X: Received signal matrix, shape (M, T).
            scan_angles: Angular grid in radians.

        Returns:
            P: Spatial spectrum, shape matching scan_angles.
        """
        # Step 1: Compute cumulant matrix
        C = compute_cumulant_matrix(X, self.rho)
        M_v = C.shape[0]

        # Step 2: Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(C)

        # Sort by eigenvalue magnitude (descending)
        idx = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Determine signal/noise subspace split
        K = self.num_sources
        if K is None:
            K = self._estimate_num_sources_from_eigenvalues(eigenvalues, M_v)

        K = min(K, M_v - 1)  # Ensure at least 1 noise eigenvector

        U_s = eigenvectors[:, :K]       # Signal subspace
        U_n = eigenvectors[:, K:]       # Noise subspace

        # Step 3: Compute spectrum for each scan angle
        P = np.zeros(len(scan_angles))

        for i, theta in enumerate(scan_angles):
            a_v = self.array.virtual_steering_vector(theta, self.rho)

            # Truncate/pad to match virtual array size
            if len(a_v) > M_v:
                a_v = a_v[:M_v]
            elif len(a_v) < M_v:
                a_v_new = np.zeros(M_v, dtype=complex)
                a_v_new[:len(a_v)] = a_v
                a_v = a_v_new

            if self.spectrum_type == "combined":
                P[i] = self._combined_spectrum(a_v, U_s, U_n)
            elif self.spectrum_type == "signal":
                P[i] = self._signal_spectrum(a_v, U_s)
            elif self.spectrum_type == "noise":
                P[i] = self._noise_spectrum(a_v, U_n)

        # Normalize spectrum
        P_max = np.max(P)
        if P_max > 0:
            P = P / P_max

        return P

    def _signal_spectrum(self, a_v, U_s):
        """Signal-subspace constrained spectrum.

        Solve: min ||w||^2  s.t. a_v^H w = 1,  U_n^H w = 0
        This constrains w to lie in the signal subspace.

        When the look direction aligns with a true DOA, the constraint
        is easily satisfied → small ||w||^2 → large 1/||w||^2.
        """
        # Project a_v onto signal subspace
        P_s = U_s @ U_s.conj().T
        a_proj = P_s @ a_v

        denom = np.abs(a_v.conj() @ a_proj) ** 2
        if denom < 1e-15:
            return 0.0
        return np.real(a_v.conj() @ a_proj) / (np.linalg.norm(a_proj) ** 2 + 1e-15)

    def _noise_spectrum(self, a_v, U_n):
        """Noise-subspace constrained spectrum (MUSIC-like).

        P(theta) = 1 / (a_v^H * U_n * U_n^H * a_v)
        Peaks where a_v is orthogonal to noise subspace.
        """
        noise_proj = U_n @ U_n.conj().T @ a_v
        denom = np.real(a_v.conj() @ noise_proj)
        if denom < 1e-15:
            return 1e10  # Peak (orthogonal to noise subspace)
        return 1.0 / denom

    def _combined_spectrum(self, a_v, U_s, U_n):
        """Combined signal × noise spectrum (sharpest peaks).

        P(theta) = (a_v^H P_s a_v) / (a_v^H P_n a_v)

        where P_s = U_s U_s^H (signal subspace projector)
              P_n = U_n U_n^H (noise subspace projector)

        This produces sharper peaks than either subspace alone because:
        - Numerator is large when aligned with signal subspace
        - Denominator is small when orthogonal to noise subspace
        """
        # Signal subspace projection
        sig_proj = U_s.conj().T @ a_v
        numerator = np.real(np.sum(np.abs(sig_proj) ** 2))

        # Noise subspace projection
        noise_proj = U_n.conj().T @ a_v
        denominator = np.real(np.sum(np.abs(noise_proj) ** 2))

        if denominator < 1e-15:
            return 1e10 * (numerator + 1e-15)

        return numerator / denominator

    def _estimate_num_sources_cumulant(self, X):
        """Estimate number of sources from cumulant eigenvalues."""
        C = compute_cumulant_matrix(X, self.rho)
        eigenvalues = np.sort(np.abs(np.linalg.eigvalsh(C)))[::-1]
        return self._estimate_num_sources_from_eigenvalues(eigenvalues, C.shape[0])

    def _estimate_num_sources_from_eigenvalues(self, eigenvalues, M_v):
        """Estimate K from eigenvalue profile using ratio test.

        Looks for the largest gap in the eigenvalue sequence.
        """
        eig_abs = np.abs(eigenvalues)
        if len(eig_abs) < 2:
            return 1

        # Ratio test: find largest drop
        ratios = eig_abs[:-1] / (eig_abs[1:] + 1e-30)
        K = np.argmax(ratios) + 1

        # Sanity check
        K = max(1, min(K, M_v - 1))
        return K
