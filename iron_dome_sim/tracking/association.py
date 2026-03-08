"""Data association algorithms for multi-target tracking.

Associates measurements to existing tracks using various strategies:
- GNN: Global Nearest Neighbor (Hungarian algorithm)
- JPDA: Joint Probabilistic Data Association
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


class GNN:
    """Global Nearest Neighbor data association.

    Uses the Hungarian algorithm to find the optimal one-to-one
    assignment between measurements and tracks that minimizes
    total Mahalanobis distance.
    """

    def __init__(self, gate_threshold=9.21):
        """
        Args:
            gate_threshold: Chi-squared gating threshold.
                           9.21 = 99% gate for 2D measurements.
        """
        self.gate_threshold = gate_threshold

    def associate(self, tracks, measurements):
        """Find optimal track-to-measurement associations.

        Args:
            tracks: List of track objects (each has filter with innovation method).
            measurements: Array of measurements, shape (N_meas, dim_obs).

        Returns:
            associations: Dict {track_idx: meas_idx} for associated pairs.
            unassociated_tracks: List of track indices without measurements.
            unassociated_meas: List of measurement indices without tracks.
        """
        N_tracks = len(tracks)
        N_meas = len(measurements)

        if N_tracks == 0 or N_meas == 0:
            return {}, list(range(N_tracks)), list(range(N_meas))

        # Build cost matrix (Mahalanobis distances)
        cost = np.full((N_tracks, N_meas), 1e10)
        gated = np.zeros((N_tracks, N_meas), dtype=bool)

        for i, track in enumerate(tracks):
            for j, z in enumerate(measurements):
                y, S = track.filter.innovation(z)
                try:
                    S_inv = np.linalg.inv(S)
                    d2 = np.real(y @ S_inv @ y)
                except np.linalg.LinAlgError:
                    d2 = 1e10

                if d2 < self.gate_threshold:
                    cost[i, j] = d2
                    gated[i, j] = True

        # Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost)

        associations = {}
        for r, c in zip(row_ind, col_ind):
            if gated[r, c]:
                associations[r] = c

        associated_tracks = set(associations.keys())
        associated_meas = set(associations.values())

        unassociated_tracks = [i for i in range(N_tracks) if i not in associated_tracks]
        unassociated_meas = [j for j in range(N_meas) if j not in associated_meas]

        return associations, unassociated_tracks, unassociated_meas


class JPDA:
    """Joint Probabilistic Data Association.

    Computes association probabilities for all measurement-to-track
    hypotheses and uses probability-weighted updates. Better for
    dense target environments with measurement ambiguity.
    """

    def __init__(self, gate_threshold=9.21, Pd=0.9, clutter_density=1e-3):
        """
        Args:
            gate_threshold: Chi-squared gating threshold.
            Pd: Detection probability.
            clutter_density: Spatial clutter density (false alarms per unit volume).
        """
        self.gate_threshold = gate_threshold
        self.Pd = Pd
        self.clutter_density = clutter_density

    def associate(self, tracks, measurements):
        """Compute JPDA association probabilities.

        Args:
            tracks: List of track objects.
            measurements: Array of measurements, shape (N_meas, dim_obs).

        Returns:
            beta: Association probability matrix, shape (N_tracks, N_meas+1).
                  beta[i, j] = P(measurement j originated from track i).
                  beta[i, -1] = P(no measurement for track i).
            unassociated_meas: Indices of measurements not gated by any track.
        """
        N_tracks = len(tracks)
        N_meas = len(measurements)

        if N_tracks == 0 or N_meas == 0:
            return np.ones((N_tracks, 1)), list(range(N_meas))

        # Compute likelihoods (gated)
        likelihood = np.zeros((N_tracks, N_meas))
        gated = np.zeros((N_tracks, N_meas), dtype=bool)

        for i, track in enumerate(tracks):
            for j, z in enumerate(measurements):
                y, S = track.filter.innovation(z)
                try:
                    S_inv = np.linalg.inv(S)
                    d2 = np.real(y @ S_inv @ y)
                except np.linalg.LinAlgError:
                    continue

                if d2 < self.gate_threshold:
                    gated[i, j] = True
                    # Gaussian likelihood
                    dim = len(y)
                    det_S = np.abs(np.linalg.det(S))
                    norm = 1.0 / ((2 * np.pi) ** (dim / 2) * np.sqrt(det_S + 1e-30))
                    likelihood[i, j] = norm * np.exp(-0.5 * d2)

        # Association probabilities (simplified marginal approach)
        beta = np.zeros((N_tracks, N_meas + 1))

        for i in range(N_tracks):
            gated_meas = np.where(gated[i])[0]

            if len(gated_meas) == 0:
                beta[i, -1] = 1.0
                continue

            # Unnormalized probabilities
            for j in gated_meas:
                beta[i, j] = self.Pd * likelihood[i, j]

            # No-detection hypothesis
            beta[i, -1] = (1 - self.Pd) * self.clutter_density

            # Normalize
            total = np.sum(beta[i])
            if total > 0:
                beta[i] /= total
            else:
                beta[i, -1] = 1.0

        # Find measurements not gated by any track
        any_gated = np.any(gated, axis=0)
        unassociated_meas = [j for j in range(N_meas) if not any_gated[j]]

        return beta, unassociated_meas

    def weighted_update(self, track, measurements, beta_i):
        """Perform probability-weighted update for a single track.

        Args:
            track: Track object with filter.
            measurements: All measurements.
            beta_i: Association probabilities for this track, shape (N_meas+1,).
        """
        N_meas = len(measurements)

        # Combined innovation (probability-weighted)
        combined_y = np.zeros(track.filter.model.dim_obs)
        for j in range(N_meas):
            if beta_i[j] > 1e-10:
                y, S = track.filter.innovation(measurements[j])
                combined_y += beta_i[j] * y

        # Update with combined innovation
        if np.linalg.norm(combined_y) > 1e-10:
            # Approximate: treat combined innovation as single measurement
            z_combined = track.filter.model.observe(track.filter.x) + combined_y
            track.filter.update(z_combined)
