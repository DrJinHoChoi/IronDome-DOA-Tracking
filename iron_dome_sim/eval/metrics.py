"""Performance evaluation metrics for DOA estimation and tracking.

Implements standard metrics used in radar signal processing and
multi-target tracking literature.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def rmse_doa(estimated, true):
    """Root Mean Square Error for DOA estimation.

    Handles different numbers of estimated vs true sources
    by using optimal assignment.

    Args:
        estimated: Estimated DOA angles in radians.
        true: True DOA angles in radians.

    Returns:
        rmse: RMSE in radians.
        assignment: Optimal pairing of estimated to true DOAs.
    """
    if len(estimated) == 0 or len(true) == 0:
        return float('inf'), []

    # Build cost matrix (angular distance)
    cost = np.zeros((len(estimated), len(true)))
    for i, est in enumerate(estimated):
        for j, tru in enumerate(true):
            cost[i, j] = _angular_distance(est, tru) ** 2

    # Optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost)

    # RMSE over assigned pairs + penalty for unmatched true sources
    total_error = sum(cost[r, c] for r, c in zip(row_ind, col_ind))
    n_pairs = len(row_ind)
    n_missed = len(true) - n_pairs

    # Penalty: unmatched true sources get max possible error (pi/2 = 90 deg)
    penalty = n_missed * (np.pi / 2) ** 2

    rmse = np.sqrt((total_error + penalty) / max(len(true), 1))
    assignment = list(zip(row_ind, col_ind))

    return rmse, assignment


def gospa(estimated_positions, true_positions, c=100, p=2, alpha=2):
    """Generalized Optimal Sub-Pattern Assignment metric.

    Standard multi-target tracking evaluation metric that accounts for:
    - Localization errors of properly detected targets
    - Missed target penalty
    - False alarm penalty

    Args:
        estimated_positions: Estimated positions, shape (N_est, D).
        true_positions: True positions, shape (N_true, D).
        c: Cutoff distance (max penalty per target).
        p: Distance order (typically 2).
        alpha: Missed/false target penalty weight.

    Returns:
        gospa_val: GOSPA metric value.
        decomposition: Dict with 'localization', 'missed', 'false' components.
    """
    n_est = len(estimated_positions)
    n_true = len(true_positions)

    if n_est == 0 and n_true == 0:
        return 0.0, {'localization': 0, 'missed': 0, 'false': 0}

    if n_est == 0:
        missed_penalty = n_true * (c ** p / alpha)
        return missed_penalty ** (1/p), {'localization': 0, 'missed': missed_penalty, 'false': 0}

    if n_true == 0:
        false_penalty = n_est * (c ** p / alpha)
        return false_penalty ** (1/p), {'localization': 0, 'missed': 0, 'false': false_penalty}

    # Cost matrix: min(d^p, c^p)
    cost = np.zeros((n_est, n_true))
    for i in range(n_est):
        for j in range(n_true):
            d = np.linalg.norm(
                np.array(estimated_positions[i]) - np.array(true_positions[j])
            )
            cost[i, j] = min(d ** p, c ** p)

    # Optimal assignment (pad if different sizes)
    if n_est <= n_true:
        row_ind, col_ind = linear_sum_assignment(cost)
    else:
        row_ind, col_ind = linear_sum_assignment(cost.T)
        row_ind, col_ind = col_ind, row_ind

    # Localization error
    localization = sum(cost[r, c] for r, c in zip(row_ind, col_ind))

    # Missed and false penalties
    n_assigned = len(row_ind)
    n_missed = max(0, n_true - n_assigned)
    n_false = max(0, n_est - n_assigned)

    missed_penalty = n_missed * (c ** p / alpha)
    false_penalty = n_false * (c ** p / alpha)

    total = localization + missed_penalty + false_penalty
    gospa_val = total ** (1/p)

    return gospa_val, {
        'localization': localization ** (1/p) if localization > 0 else 0,
        'missed': missed_penalty ** (1/p) if missed_penalty > 0 else 0,
        'false': false_penalty ** (1/p) if false_penalty > 0 else 0,
        'n_assigned': n_assigned,
        'n_missed': n_missed,
        'n_false': n_false,
    }


def track_purity(track_assignments, true_target_ids):
    """Compute track purity (fraction of correct associations).

    Args:
        track_assignments: Dict {track_id: list of assigned target_ids over time}.
        true_target_ids: Dict {track_id: true target_id}.

    Returns:
        purity: Overall track purity (0-1).
        per_track: Dict {track_id: purity}.
    """
    if len(track_assignments) == 0:
        return 0.0, {}

    per_track = {}
    total_correct = 0
    total_assignments = 0

    for track_id, assignments in track_assignments.items():
        if len(assignments) == 0:
            per_track[track_id] = 0.0
            continue

        true_id = true_target_ids.get(track_id)
        if true_id is None:
            per_track[track_id] = 0.0
            continue

        correct = sum(1 for a in assignments if a == true_id)
        per_track[track_id] = correct / len(assignments)
        total_correct += correct
        total_assignments += len(assignments)

    purity = total_correct / max(total_assignments, 1)
    return purity, per_track


def crlb_doa(M, K, snr_db, T, d=0.5, rho=1):
    """Cramér-Rao Lower Bound for DOA estimation.

    Computes the theoretical minimum variance for unbiased DOA
    estimation with a ULA.

    Args:
        M: Number of sensors.
        K: Number of sources.
        snr_db: SNR in dB.
        T: Number of snapshots.
        d: Inter-element spacing in wavelengths.
        rho: Order parameter (1=standard, >1=higher-order).

    Returns:
        crlb: CRLB variance for each source DOA (radians^2).
    """
    snr = 10 ** (snr_db / 10)

    # Effective array size for higher-order processing
    M_eff = rho * (M - 1) + 1 if rho > 1 else M

    # Simplified CRLB for well-separated sources
    # CRLB ≈ 6 / (T * snr * M_eff * (M_eff^2 - 1) * (2*pi*d)^2)
    crlb_var = 6.0 / (
        T * snr * M_eff * (M_eff ** 2 - 1) * (2 * np.pi * d) ** 2
    )

    return np.full(K, crlb_var)


def detection_rate(estimated, true, threshold_rad=0.05):
    """Compute detection rate (probability of detection).

    A true source is "detected" if there is an estimated DOA
    within threshold_rad of it.

    Args:
        estimated: Estimated DOAs in radians.
        true: True DOAs in radians.
        threshold_rad: Detection threshold in radians (~3 degrees).

    Returns:
        pd: Probability of detection.
        pfa: Probability of false alarm.
    """
    if len(true) == 0:
        pfa = len(estimated)  # All are false alarms
        return 0.0, pfa

    detected = 0
    used_est = set()

    for t in true:
        for i, e in enumerate(estimated):
            if i not in used_est and _angular_distance(e, t) < threshold_rad:
                detected += 1
                used_est.add(i)
                break

    pd = detected / len(true)
    n_false = len(estimated) - len(used_est)
    pfa = n_false / max(len(estimated), 1)

    return pd, pfa


def _angular_distance(a1, a2):
    """Compute angular distance handling wrap-around."""
    d = abs(a1 - a2)
    return min(d, 2 * np.pi - d)
