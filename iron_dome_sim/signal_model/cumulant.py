"""Higher-order cumulant matrix computation for underdetermined DOA.

Implements the 2rho-th order cumulant matrix using the SUM co-array
structure (x ⊗ x, not x ⊗ x*) to achieve virtual aperture extension.

Key insight: For ULA with M sensors, the 4th-order cumulant (rho=2)
creates a virtual array with lags 0 to 2(M-1), giving a Toeplitz
matrix of size (2(M-1)+1) that can resolve up to 2(M-1) sources
using subspace methods — more than the M-1 limit of 2nd-order methods.

Gaussian noise vanishes in cumulants of order > 2 (noise elimination).

Reference:
    Choi & Yoo, IEEE TSP 2015.
"""

import numpy as np
from scipy.linalg import toeplitz


def compute_cumulant_matrix(X, rho=2):
    """Compute the noise-eliminated 2rho-th order cumulant matrix.

    Uses the sum co-array structure for virtual aperture extension.

    Args:
        X: Received signal matrix, shape (M, T).
        rho: Order parameter (rho >= 1).

    Returns:
        C: Hermitian Toeplitz cumulant matrix, shape (M_v, M_v)
           where M_v = rho*(M-1)+1.
    """
    M, T = X.shape

    if rho == 1:
        return X @ X.conj().T / T

    elif rho == 2:
        return _compute_4th_order_cumulant_toeplitz(X)

    else:
        return _compute_higher_order_cumulant_toeplitz(X, rho)


def _compute_4th_order_cumulant_toeplitz(X):
    """Compute 4th-order cumulant as Toeplitz matrix via sum co-array.

    The 4th-order cumulant for circular signals:
        c4(i1,i2,i3,i4) = E[x_i1 x_i2 x*_i3 x*_i4]
                         - E[x_i1 x*_i3]E[x_i2 x*_i4]
                         - E[x_i1 x*_i4]E[x_i2 x*_i3]

    For ULA, c4 depends only on lag τ = (i1+i2)-(i3+i4).
    We compute c4(τ) for τ = 0, 1, ..., 2(M-1) and form a Toeplitz matrix.

    Args:
        X: Received signal matrix, shape (M, T).

    Returns:
        C4_toeplitz: Hermitian Toeplitz matrix, shape (M_v, M_v).
    """
    M, T = X.shape
    L = 2 * (M - 1)  # Maximum lag
    M_v = L + 1       # Virtual array size

    # Covariance matrix
    R = X @ X.conj().T / T

    # Compute cumulant for each lag τ by averaging over all
    # (i1,i2,i3,i4) combinations with (i1+i2)-(i3+i4) = τ
    c4_lags = np.zeros(M_v, dtype=complex)
    c4_counts = np.zeros(M_v)

    for i1 in range(M):
        for i2 in range(M):
            for i3 in range(M):
                for i4 in range(M):
                    tau = (i1 + i2) - (i3 + i4)
                    if 0 <= tau <= L:
                        # 4th-order moment
                        m4 = np.mean(X[i1] * X[i2] * X[i3].conj() * X[i4].conj())
                        # Subtract Gaussian terms
                        g1 = R[i1, i3] * R[i2, i4]
                        g2 = R[i1, i4] * R[i2, i3]
                        c4 = m4 - g1 - g2

                        c4_lags[tau] += c4
                        c4_counts[tau] += 1

    # Average
    mask = c4_counts > 0
    c4_lags[mask] /= c4_counts[mask]

    # Form Hermitian Toeplitz matrix
    # T[i,j] = c4(i-j) for i >= j, c4(j-i)^* for i < j
    first_col = c4_lags
    first_row = c4_lags.conj()
    C4 = toeplitz(first_col, first_row)

    return C4


def _compute_higher_order_cumulant_toeplitz(X, rho):
    """Compute 2rho-th order cumulant Toeplitz matrix.

    For rho > 2, uses the rho-fold sum co-array.
    The lag τ = (sum of rho 'row' indices) - (sum of rho 'col' indices)
    ranges from 0 to rho*(M-1).

    Args:
        X: Received signal matrix, shape (M, T).
        rho: Order parameter.

    Returns:
        C_toeplitz: Hermitian Toeplitz matrix, shape (M_v, M_v).
    """
    M, T = X.shape
    L = rho * (M - 1)
    M_v = L + 1

    # For computational tractability with higher orders,
    # use the Kronecker product approach with sum co-array extraction.

    # Build z(t) = x(t) ⊗ x(t) ⊗ ... (rho times, NO conjugate)
    # Then C_{2rho} ~ E[z z^H] - Gaussian terms

    # Compute the sum co-array lags directly
    # This avoids forming the full M^rho x M^rho matrix

    # For each lag τ, accumulate contributions
    c_lags = np.zeros(M_v, dtype=complex)
    c_counts = np.zeros(M_v)

    # Sample-based estimation of the 2rho-th order cumulant
    # Using the dominant term: M_{2rho} (moment)
    # minus the major Gaussian partition (R ⊗ R ⊗ ... ⊗ R)

    # Compute moments via sum co-array
    # For each time snapshot, compute the rho-fold product contributions
    for tau in range(M_v):
        moment_sum = 0.0
        count = 0

        # Find all index combinations where sum(row) - sum(col) = tau
        # Use a sampling approach for large rho
        _accumulate_cumulant_lag(X, R=X @ X.conj().T / T,
                                rho=rho, M=M, tau=tau,
                                result=c_lags, counts=c_counts)

    mask = c_counts > 0
    c_lags[mask] /= c_counts[mask]

    first_col = c_lags
    first_row = c_lags.conj()
    return toeplitz(first_col, first_row)


def _accumulate_cumulant_lag(X, R, rho, M, tau, result, counts):
    """Accumulate cumulant values for a specific lag using direct computation.

    For rho=3 (6th order), iterates over all valid index combinations.
    """
    T = X.shape[1]

    if rho == 3:
        # 6th order: iterate i1,i2,i3,j1,j2,j3
        # where (i1+i2+i3) - (j1+j2+j3) = tau
        for i1 in range(M):
            for i2 in range(M):
                for i3 in range(M):
                    row_sum = i1 + i2 + i3
                    col_sum_needed = row_sum - tau
                    if col_sum_needed < 0 or col_sum_needed > 3 * (M - 1):
                        continue
                    for j1 in range(M):
                        for j2 in range(M):
                            j3 = col_sum_needed - j1 - j2
                            if 0 <= j3 < M:
                                # 6th-order moment
                                m6 = np.mean(
                                    X[i1] * X[i2] * X[i3] *
                                    X[j1].conj() * X[j2].conj() * X[j3].conj()
                                )
                                # Subtract dominant Gaussian terms
                                g = (R[i1, j1] * R[i2, j2] * R[i3, j3] +
                                     R[i1, j1] * R[i2, j3] * R[i3, j2] +
                                     R[i1, j2] * R[i2, j1] * R[i3, j3] +
                                     R[i1, j2] * R[i2, j3] * R[i3, j1] +
                                     R[i1, j3] * R[i2, j1] * R[i3, j2] +
                                     R[i1, j3] * R[i2, j2] * R[i3, j1])
                                c6 = m6 - g
                                result[tau] += c6
                                counts[tau] += 1
