"""Exact Cramér-Rao Lower Bound (CRLB) for DOA estimation.

Computes the Fisher Information Matrix (FIM) and its inverse (CRLB)
for both standard covariance-based and higher-order cumulant-based
DOA estimation.

Key formulas:
    Standard (rho=1): Stochastic CRB [Stoica & Nehorai, IEEE TASSP 1989]
        CRB(theta) = 1/(2T) * [Re(D^H P_A_perp D ⊙ (R_s A^H R^{-1} A R_s)^T)]^{-1}

    Higher-order (rho>1): Virtual array CRB
        Uses virtual steering matrix A_v (size M_v × K) from rho-th order cumulant,
        with effective SNR gain from Gaussian noise elimination.

References:
    [1] P. Stoica and A. Nehorai, "MUSIC, Maximum Likelihood, and
        Cramer-Rao Bound," IEEE Trans. ASSP, vol. 37, no. 5, 1989.
    [2] P. Stoica and A. Nehorai, "Performance study of conditional
        and unconditional direction-of-arrival estimation," IEEE Trans.
        ASSP, vol. 38, no. 10, 1990.
    [3] Choi & Yoo, IEEE TSP, 2015 (COP algorithm context).
"""

import numpy as np


def steering_vector_ula(theta, M, d=0.5):
    """ULA steering vector.

    Args:
        theta: DOA angle in radians.
        M: Number of sensors.
        d: Inter-element spacing in wavelengths.

    Returns:
        a: Steering vector, shape (M,).
    """
    n = np.arange(M)
    return np.exp(1j * 2 * np.pi * d * n * np.sin(theta))


def steering_derivative_ula(theta, M, d=0.5):
    """Derivative of ULA steering vector w.r.t. theta.

    d(a)/d(theta) = j * 2*pi*d * cos(theta) * diag(n) * a(theta)

    Args:
        theta: DOA angle in radians.
        M: Number of sensors.
        d: Inter-element spacing in wavelengths.

    Returns:
        da: Derivative vector, shape (M,).
    """
    n = np.arange(M)
    a = steering_vector_ula(theta, M, d)
    return 1j * 2 * np.pi * d * np.cos(theta) * n * a


def virtual_steering_vector(theta, M, d=0.5, rho=2):
    """Virtual steering vector for rho-th order cumulant.

    The virtual array from the rho-th order cumulant has
    M_v = rho*(M-1)+1 elements with extended aperture.

    Args:
        theta: DOA angle in radians.
        M: Physical number of sensors.
        d: Inter-element spacing.
        rho: Cumulant order parameter.

    Returns:
        a_v: Virtual steering vector, shape (M_v,).
    """
    M_v = rho * (M - 1) + 1
    n_v = np.arange(M_v)
    return np.exp(1j * 2 * np.pi * d * n_v * np.sin(theta))


def virtual_steering_derivative(theta, M, d=0.5, rho=2):
    """Derivative of virtual steering vector w.r.t. theta.

    Args:
        theta: DOA angle in radians.
        M: Physical number of sensors.
        d: Inter-element spacing.
        rho: Cumulant order parameter.

    Returns:
        da_v: Derivative vector, shape (M_v,).
    """
    M_v = rho * (M - 1) + 1
    n_v = np.arange(M_v)
    a_v = virtual_steering_vector(theta, M, d, rho)
    return 1j * 2 * np.pi * d * np.cos(theta) * n_v * a_v


def crlb_stochastic(thetas, M, snr_db, T, d=0.5):
    """Stochastic CRB for standard covariance-based DOA estimation.

    Assumes unconditional (stochastic) signal model:
        X = A * S + N,  S ~ CN(0, R_s),  N ~ CN(0, sigma^2 * I)

    CRB = 1/(2T) * [Re(D^H P_A_perp D  ⊙  (R_s A^H R^{-1} A R_s)^T)]^{-1}

    Args:
        thetas: True DOA angles in radians, shape (K,).
        M: Number of sensors.
        snr_db: SNR in dB (per source).
        T: Number of snapshots.
        d: Inter-element spacing in wavelengths.

    Returns:
        crb: CRB variance per DOA, shape (K,) in radians^2.
        fim: Fisher Information Matrix, shape (K, K).
    """
    thetas = np.asarray(thetas)
    K = len(thetas)
    snr = 10 ** (snr_db / 10)
    sigma2 = 1.0  # Noise power normalized to 1
    signal_power = snr * sigma2  # Per-source power

    # Steering matrix A (M x K)
    A = np.column_stack([steering_vector_ula(t, M, d) for t in thetas])

    # Derivative matrix D (M x K)
    D = np.column_stack([steering_derivative_ula(t, M, d) for t in thetas])

    # Source covariance R_s = diag(signal_power) (equal power)
    R_s = signal_power * np.eye(K)

    # Data covariance R = A R_s A^H + sigma^2 I
    R = A @ R_s @ A.conj().T + sigma2 * np.eye(M)

    # Noise subspace projector P_A_perp = I - A (A^H A)^{-1} A^H
    AHA = A.conj().T @ A
    try:
        AHA_inv = np.linalg.inv(AHA)
    except np.linalg.LinAlgError:
        AHA_inv = np.linalg.pinv(AHA)
    P_perp = np.eye(M) - A @ AHA_inv @ A.conj().T

    # D^H P_perp D  (K x K)
    DHP = D.conj().T @ P_perp @ D

    # R_s A^H R^{-1} A R_s  (K x K)
    R_inv = np.linalg.inv(R)
    inner = R_s @ A.conj().T @ R_inv @ A @ R_s

    # FIM = 2T * Re(DHP ⊙ inner^T)
    fim = 2 * T * np.real(DHP * inner.T)

    # CRB = FIM^{-1}
    try:
        crb_matrix = np.linalg.inv(fim)
        crb = np.diag(crb_matrix)
    except np.linalg.LinAlgError:
        crb = np.full(K, np.inf)

    # Ensure non-negative
    crb = np.maximum(crb, 0)

    return crb, fim


def crlb_cop(thetas, M, snr_db, T, d=0.5, rho=2):
    """CRB for higher-order cumulant-based DOA estimation (COP).

    Uses the virtual array manifold from rho-th order cumulants.
    Key differences from standard CRB:
    1. Virtual array size M_v = rho*(M-1)+1 (extended aperture)
    2. Gaussian noise eliminated in cumulants (effective noise-free)
    3. Cumulant estimation variance scales as ~1/T

    The virtual array CRB follows the same structure as standard CRB
    but with A replaced by A_v and adjusted effective SNR.

    For the 4th-order cumulant (rho=2):
    - The cumulant-domain "SNR" is approximately snr^2 (signal cumulant
      grows as power^2, noise cumulant is zero for Gaussian)
    - But cumulant estimation variance is ~6/T for 4th-order
    - Net effect: CRB uses M_v, snr^rho, and T/C_rho scaling

    Args:
        thetas: True DOA angles in radians, shape (K,).
        M: Number of physical sensors.
        snr_db: SNR in dB (per source).
        T: Number of snapshots.
        d: Inter-element spacing in wavelengths.
        rho: Cumulant order parameter (default: 2 for 4th-order).

    Returns:
        crb: CRB variance per DOA, shape (K,) in radians^2.
        fim: Fisher Information Matrix, shape (K, K).
    """
    thetas = np.asarray(thetas)
    K = len(thetas)
    M_v = rho * (M - 1) + 1

    snr = 10 ** (snr_db / 10)
    signal_power = snr  # Normalized noise power = 1

    # Cumulant-domain effective parameters:
    # 4th-order cumulant of signal: kappa_4 ~ signal_power^rho
    # Noise cumulant (Gaussian) = 0 (eliminated!)
    # Cumulant estimation variance factor: C_rho
    # For rho=2 (4th-order): C_rho ≈ 6 (higher variance)
    # For rho=3 (6th-order): C_rho ≈ 120
    cumulant_variance_factor = {1: 1, 2: 6, 3: 120}.get(rho, rho * (rho + 1))

    # Effective SNR in cumulant domain
    # Signal contribution: signal_power^rho
    # Effective noise: cumulant estimation error ~ C_rho / T
    effective_signal_power = signal_power ** rho

    # Virtual steering matrix A_v (M_v x K)
    A_v = np.column_stack([virtual_steering_vector(t, M, d, rho) for t in thetas])

    # Virtual derivative matrix D_v (M_v x K)
    D_v = np.column_stack([virtual_steering_derivative(t, M, d, rho) for t in thetas])

    # Source covariance in cumulant domain
    R_s = effective_signal_power * np.eye(K)

    # In cumulant domain, noise is eliminated (Gaussian noise has zero
    # higher-order cumulants). The "noise" is cumulant estimation error.
    # Effective noise variance: sigma_c^2 ≈ C_rho * sigma^(2*rho) / T
    # For normalized noise (sigma=1): sigma_c^2 ≈ C_rho / T
    sigma_c2 = cumulant_variance_factor / T

    # Data "covariance" in cumulant domain
    R_c = A_v @ R_s @ A_v.conj().T + sigma_c2 * np.eye(M_v)

    # Noise subspace projector
    AvHAv = A_v.conj().T @ A_v
    try:
        AvHAv_inv = np.linalg.inv(AvHAv)
    except np.linalg.LinAlgError:
        AvHAv_inv = np.linalg.pinv(AvHAv)
    P_perp = np.eye(M_v) - A_v @ AvHAv_inv @ A_v.conj().T

    # D_v^H P_perp D_v
    DHP = D_v.conj().T @ P_perp @ D_v

    # Inner product for FIM
    R_c_inv = np.linalg.inv(R_c)
    inner = R_s @ A_v.conj().T @ R_c_inv @ A_v @ R_s

    # Effective number of "snapshots" for cumulant estimation
    # For 4th-order cumulant, the effective T is approximately T
    # (though with higher variance captured by C_rho)
    T_eff = T

    # FIM = 2 * T_eff * Re(DHP ⊙ inner^T)
    fim = 2 * T_eff * np.real(DHP * inner.T)

    # CRB = FIM^{-1}
    try:
        crb_matrix = np.linalg.inv(fim)
        crb = np.diag(crb_matrix)
    except np.linalg.LinAlgError:
        crb = np.full(K, np.inf)

    crb = np.maximum(crb, 0)

    return crb, fim


def crlb_rmse(thetas, M, snr_db_range, T, d=0.5, rho=1):
    """Compute CRLB RMSE curve over an SNR range.

    Convenience function for plotting CRLB alongside RMSE benchmarks.

    Args:
        thetas: True DOA angles in radians.
        M: Number of sensors.
        snr_db_range: Array of SNR values in dB.
        T: Number of snapshots.
        d: Inter-element spacing.
        rho: 1 for standard, 2 for COP (4th-order).

    Returns:
        crlb_rmse_deg: CRLB RMSE in degrees for each SNR, shape (len(snr_db_range),).
    """
    thetas = np.asarray(thetas)
    crlb_fn = crlb_cop if rho > 1 else crlb_stochastic
    rmse_list = []

    for snr_db in snr_db_range:
        if rho > 1:
            crb, _ = crlb_fn(thetas, M, snr_db, T, d, rho)
        else:
            crb, _ = crlb_fn(thetas, M, snr_db, T, d)

        # RMSE = sqrt(mean(CRB_diag))
        mean_crb = np.mean(crb)
        if mean_crb < np.inf and mean_crb >= 0:
            rmse_list.append(np.degrees(np.sqrt(mean_crb)))
        else:
            rmse_list.append(np.inf)

    return np.array(rmse_list)
