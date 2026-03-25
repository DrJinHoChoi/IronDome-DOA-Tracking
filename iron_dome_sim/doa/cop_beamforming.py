"""COP-based beamforming for DOA estimation without knowing K.

COP-CBF: Conventional Beamforming on virtual array (cumulant domain)
COP-MVDR: MVDR/Capon beamforming on cumulant matrix

Key advantage over subspace COP: no need to estimate K.
Key advantage over standard beamforming: virtual aperture extension
    M=4 physical → M_v=7 virtual (rho=2), resolution doubles.

Reference:
    Choi & Yoo, IEEE TSP 2015.
"""

import numpy as np
from .base import DOAEstimator
from .spectrum import find_peaks_doa
from ..signal_model.cumulant import compute_cumulant_matrix


class COP_CBF(DOAEstimator):
    """Conventional Beamforming on COP virtual array.

    P_CBF(θ) = a_v^H(θ) C a_v(θ)

    No K estimation required. Resolution limited by virtual aperture
    but still better than physical-array CBF.
    """

    def __init__(self, array, num_sources=None, rho=2):
        super().__init__(array, num_sources)
        self.rho = rho
        self.name = f"COP-CBF(ρ={rho})"

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
        C = compute_cumulant_matrix(X, rho=self.rho)
        M_v = C.shape[0]

        P = np.zeros(len(scan_angles))
        for i, theta in enumerate(scan_angles):
            a_v = self.array.virtual_steering_vector(theta, self.rho)
            P[i] = np.real(a_v.conj() @ C @ a_v)

        P = np.abs(P)
        P_max = np.max(P)
        if P_max > 0:
            P /= P_max
        return P

    @property
    def is_underdetermined(self):
        return True

    @property
    def max_sources(self):
        return self.rho * (self.array.M - 1)


class COP_MVDR(DOAEstimator):
    """MVDR (Capon) beamforming on COP cumulant matrix.

    P_MVDR(θ) = 1 / (a_v^H(θ) C^{-1} a_v(θ))

    Combines MVDR adaptive nulling with COP virtual aperture.
    No K estimation required. Better resolution than COP-CBF.
    """

    def __init__(self, array, num_sources=None, rho=2, diagonal_loading=1e-6):
        super().__init__(array, num_sources)
        self.rho = rho
        self.diagonal_loading = diagonal_loading
        self.name = f"COP-MVDR(ρ={rho})"

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
        C = compute_cumulant_matrix(X, rho=self.rho)
        M_v = C.shape[0]

        # Diagonal loading for numerical stability
        C += self.diagonal_loading * np.eye(M_v)
        C_inv = np.linalg.inv(C)

        P = np.zeros(len(scan_angles))
        for i, theta in enumerate(scan_angles):
            a_v = self.array.virtual_steering_vector(theta, self.rho)
            P[i] = 1.0 / np.real(a_v.conj() @ C_inv @ a_v)

        P = np.abs(P)
        P_max = np.max(P)
        if P_max > 0:
            P /= P_max
        return P

    @property
    def is_underdetermined(self):
        return True

    @property
    def max_sources(self):
        return self.rho * (self.array.M - 1)


class CBF(DOAEstimator):
    """Conventional Beamforming (Delay-and-Sum) on physical array.

    P_CBF(θ) = a^H(θ) R a(θ)

    Baseline with lowest resolution. K not required.
    """

    def __init__(self, array, num_sources=None):
        super().__init__(array, num_sources)
        self.name = "CBF"

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

        P = np.zeros(len(scan_angles))
        for i, theta in enumerate(scan_angles):
            a = self.array.steering_vector(theta)
            P[i] = np.real(a.conj() @ R @ a)

        P_max = np.max(P)
        if P_max > 0:
            P /= P_max
        return P
