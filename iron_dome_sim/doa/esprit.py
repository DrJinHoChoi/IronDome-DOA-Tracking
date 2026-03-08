"""ESPRIT DOA estimation algorithm.

Estimation of Signal Parameters via Rotational Invariance Techniques.
Exploits the shift-invariance structure of ULA. Limited to K < M.
"""

import numpy as np
from .base import DOAEstimator


class ESPRIT(DOAEstimator):
    """ESPRIT algorithm for DOA estimation.

    Uses the rotational invariance property of ULA subarrays
    to estimate DOA without spectral search.
    """

    def __init__(self, array, num_sources=None):
        super().__init__(array, num_sources)
        self.name = "ESPRIT"

    @property
    def is_underdetermined(self):
        return False

    def estimate(self, X, scan_angles=None):
        M, T = X.shape

        # Sample covariance
        R = X @ X.conj().T / T

        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(R)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]

        K = self.num_sources
        if K is None:
            K = self._estimate_num_sources(X)
        K = min(K, M - 1)

        # Signal subspace
        U_s = eigenvectors[:, :K]

        # Two subarrays (shift by 1)
        U1 = U_s[:-1, :]  # first M-1 rows
        U2 = U_s[1:, :]   # last M-1 rows

        # Rotation matrix: U2 = U1 * Phi
        Phi = np.linalg.lstsq(U1, U2, rcond=None)[0]

        # DOA from eigenvalues of Phi
        eig_phi = np.linalg.eigvals(Phi)
        doa_estimates = np.arcsin(np.angle(eig_phi) / (2 * np.pi * self.array.d))

        # Filter valid angles
        valid = np.abs(doa_estimates) <= np.pi / 2
        doa_estimates = np.sort(np.real(doa_estimates[valid]))

        return doa_estimates, None

    def spectrum(self, X, scan_angles):
        # ESPRIT doesn't produce a spatial spectrum
        return None
