"""Capon (MVDR) beamformer for DOA estimation.

Minimum Variance Distortionless Response beamformer.
Limited resolution compared to subspace methods.
"""

import numpy as np
from .base import DOAEstimator
from .spectrum import find_peaks_doa


class Capon(DOAEstimator):
    """Capon/MVDR beamformer for DOA estimation."""

    def __init__(self, array, num_sources=None, diagonal_loading=1e-6):
        super().__init__(array, num_sources)
        self.diagonal_loading = diagonal_loading
        self.name = "Capon"

    def estimate(self, X, scan_angles=None):
        if scan_angles is None:
            scan_angles = self._default_scan_angles()

        P = self.spectrum(X, scan_angles)

        K = self.num_sources
        if K is None:
            K = self._estimate_num_sources(X)

        doa_estimates = find_peaks_doa(P, scan_angles, K)
        return doa_estimates, P

    def spectrum(self, X, scan_angles):
        M, T = X.shape
        R = X @ X.conj().T / T
        R += self.diagonal_loading * np.eye(M)  # regularization
        R_inv = np.linalg.inv(R)

        P = np.zeros(len(scan_angles))
        for i, theta in enumerate(scan_angles):
            a = self.array.steering_vector(theta)
            P[i] = 1.0 / np.real(a.conj() @ R_inv @ a)

        P_max = np.max(np.abs(P))
        if P_max > 0:
            P = np.abs(P) / P_max
        return P
