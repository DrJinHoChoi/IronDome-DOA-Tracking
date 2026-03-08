"""Sparse recovery based DOA estimation (L1-SVD, LASSO).

These methods can handle underdetermined scenarios (K > M) by exploiting
spatial sparsity via L1 minimization. However, they are typically slower
and less accurate than the proposed COP algorithm.
"""

import numpy as np
from scipy.optimize import minimize
from .base import DOAEstimator
from .spectrum import find_peaks_doa


class L1SVD(DOAEstimator):
    """L1-SVD DOA estimation via sparse signal recovery.

    Formulates DOA estimation as a sparse recovery problem:
        min ||s||_1  s.t. ||X - A_grid * S||_F < epsilon

    Can handle underdetermined case (K > M).
    """

    def __init__(self, array, num_sources=None, grid_size=361, reg_param=0.1):
        super().__init__(array, num_sources)
        self.grid_size = grid_size
        self.reg_param = reg_param
        self.name = "L1-SVD"

    @property
    def is_underdetermined(self):
        return True

    @property
    def max_sources(self):
        return self.grid_size - 1

    def estimate(self, X, scan_angles=None):
        if scan_angles is None:
            scan_angles = np.linspace(-np.pi / 2, np.pi / 2, self.grid_size)

        P = self.spectrum(X, scan_angles)

        K = self.num_sources
        if K is None:
            K = self._estimate_num_sources(X)

        doa_estimates = find_peaks_doa(P, scan_angles, K)
        return doa_estimates, P

    def spectrum(self, X, scan_angles):
        M, T = X.shape

        # Over-complete steering matrix
        A_grid = self.array.steering_matrix(scan_angles)
        G = len(scan_angles)

        # SVD of X to reduce dimensionality
        U, S_vals, Vh = np.linalg.svd(X, full_matrices=False)

        # Use dominant singular vectors
        r = min(M, T, max(self.num_sources or M, 3))
        X_reduced = U[:, :r] @ np.diag(S_vals[:r])

        # L1 minimization via LASSO formulation
        # min ||X_reduced - A_grid @ S_grid||_F^2 + lambda * ||S_grid||_1
        P = np.zeros(G)

        for col in range(r):
            x_col = X_reduced[:, col]

            # Solve via iterative reweighted least squares (IRLS)
            s = self._solve_l1(A_grid, x_col)
            P += np.abs(s) ** 2

        # Normalize
        P_max = np.max(P)
        if P_max > 0:
            P /= P_max

        return P

    def _solve_l1(self, A, x, max_iter=50):
        """Solve L1-regularized least squares via IRLS.

        min ||x - As||^2 + lambda * ||s||_1
        """
        G = A.shape[1]
        s = np.linalg.lstsq(A, x, rcond=None)[0]
        eps = 1e-6

        for _ in range(max_iter):
            # Weights from current estimate
            W = np.diag(1.0 / (np.abs(s) + eps))
            # Weighted least squares
            AHA = A.conj().T @ A + self.reg_param * W
            AHx = A.conj().T @ x
            s = np.linalg.solve(AHA, AHx)

        return s


class LASSO_DOA(DOAEstimator):
    """LASSO-based DOA estimation.

    Uses coordinate descent for L1-regularized regression
    on the over-complete steering matrix.
    """

    def __init__(self, array, num_sources=None, grid_size=361, alpha=0.1):
        super().__init__(array, num_sources)
        self.grid_size = grid_size
        self.alpha = alpha
        self.name = "LASSO"

    @property
    def is_underdetermined(self):
        return True

    def estimate(self, X, scan_angles=None):
        if scan_angles is None:
            scan_angles = np.linspace(-np.pi / 2, np.pi / 2, self.grid_size)

        P = self.spectrum(X, scan_angles)

        K = self.num_sources
        if K is None:
            K = self._estimate_num_sources(X)

        doa_estimates = find_peaks_doa(P, scan_angles, K)
        return doa_estimates, P

    def spectrum(self, X, scan_angles):
        M, T = X.shape
        A_grid = self.array.steering_matrix(scan_angles)
        G = len(scan_angles)

        P = np.zeros(G)

        for t in range(min(T, 50)):  # Use subset for speed
            x = X[:, t]
            s = self._coordinate_descent(A_grid, x)
            P += np.abs(s) ** 2

        P_max = np.max(P)
        if P_max > 0:
            P /= P_max
        return P

    def _coordinate_descent(self, A, x, max_iter=100):
        """Coordinate descent for complex LASSO."""
        M, G = A.shape
        s = np.zeros(G, dtype=complex)
        residual = x.copy()

        for _ in range(max_iter):
            for j in range(G):
                a_j = A[:, j]
                # Add back contribution of j-th component
                residual += a_j * s[j]
                # Compute correlation
                rho_j = a_j.conj() @ residual
                norm_j = np.real(a_j.conj() @ a_j)
                # Soft thresholding
                s[j] = _soft_threshold(rho_j, self.alpha) / (norm_j + 1e-10)
                # Update residual
                residual -= a_j * s[j]

        return s


def _soft_threshold(x, threshold):
    """Complex soft thresholding operator."""
    mag = np.abs(x)
    if mag <= threshold:
        return 0.0
    return x * (1.0 - threshold / mag)
