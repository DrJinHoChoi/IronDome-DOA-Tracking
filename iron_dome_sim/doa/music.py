"""MUSIC (Multiple Signal Classification) DOA estimation.

Classical subspace method using 2nd-order covariance matrix.
Limited to K < M (overdetermined case only).
"""

import numpy as np
from .base import DOAEstimator
from .spectrum import find_peaks_doa


class MUSIC(DOAEstimator):
    """MUSIC algorithm for DOA estimation.

    Standard subspace method that decomposes the covariance matrix
    into signal and noise subspaces. Cannot handle underdetermined
    scenarios (K >= M).
    """

    def __init__(self, array, num_sources=None):
        super().__init__(array, num_sources)
        self.name = "MUSIC"

    @property
    def is_underdetermined(self):
        return False

    @property
    def max_sources(self):
        return self.array.M - 1

    def estimate(self, X, scan_angles=None):
        if scan_angles is None:
            scan_angles = self._default_scan_angles()

        P = self.spectrum(X, scan_angles)

        K = self.num_sources
        if K is None:
            K = self._estimate_num_sources(X)
        K = min(K, self.max_sources)

        doa_estimates = find_peaks_doa(P, scan_angles, K)
        return doa_estimates, P

    def spectrum(self, X, scan_angles):
        M, T = X.shape

        # Sample covariance matrix
        R = X @ X.conj().T / T

        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(R)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]

        K = self.num_sources
        if K is None:
            K = self._estimate_num_sources(X)
        K = min(K, M - 1)

        # Noise subspace
        U_n = eigenvectors[:, K:]

        # MUSIC spectrum
        P = np.zeros(len(scan_angles))
        for i, theta in enumerate(scan_angles):
            a = self.array.steering_vector(theta)
            noise_proj = U_n.conj().T @ a
            denom = np.real(np.sum(np.abs(noise_proj) ** 2))
            P[i] = 1.0 / (denom + 1e-15)

        P_max = np.max(P)
        if P_max > 0:
            P /= P_max
        return P
