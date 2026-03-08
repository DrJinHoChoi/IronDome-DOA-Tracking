"""COP-RFS: Gaussian Mixture PHD Filter with COP Spectrum Birth Intensity.

Novel integration of the 2rho-th order Subspace COP DOA estimator with
the Gaussian Mixture Probability Hypothesis Density (GM-PHD) filter.

Key innovations:
    1. COP spectrum as adaptive birth intensity for PHD filter
    2. Physics-based measurement identification BEFORE update:
       - Predict track positions using inertia (constant velocity)
       - Associate COP measurements to predicted positions (Hungarian)
       - Update each track ONLY with its matched measurement
       - Birth only from unassociated measurements
    3. Velocity-gated merge to preserve tracks through crossings

Pipeline per scan:
    1. COP DOA estimation -> spectrum + measurements
    2. PREDICT existing tracks (physics/inertia)
    3. IDENTIFY: match measurements to predictions (Hungarian)
    4. UPDATE confirmed tracks with matched measurements (Kalman)
    5. BIRTH from unassociated measurements only
    6. Prune & Merge (velocity-gated)
    7. Extract state estimates

Reference:
    [1] Choi & Yoo, IEEE TSP 2015 (base COP algorithm)
    [2] Vo & Ma, IEEE TSP 2006 (GM-PHD filter)
"""

import numpy as np
from copy import deepcopy


class GaussianComponent:
    """Single Gaussian component in the GM-PHD mixture.

    Represents a weighted Gaussian: w * N(x; m, P)
    """

    __slots__ = ['weight', 'mean', 'covariance', 'label']

    def __init__(self, weight, mean, covariance, label=-1):
        self.weight = weight
        self.mean = np.asarray(mean, dtype=float).copy()
        self.covariance = np.asarray(covariance, dtype=float).copy()
        self.label = label


class COPPHD:
    """COP-RFS: GM-PHD Filter with Physics-Based Track Identification.

    Novel contribution: measurement-to-track identification using
    physics-based prediction (inertia) BEFORE the Kalman update step.
    This ensures that each track is updated only with its correct
    measurement, preventing identity confusion at target crossings.

    Pipeline per scan:
        1. COP DOA estimation -> spectrum + measurements
        2. Predict existing tracks forward (constant velocity / inertia)
        3. Associate measurements to predicted positions (Hungarian)
        4. Update confirmed tracks with matched measurements (Kalman)
        5. Birth from unassociated measurements (COP spectrum weighted)
        6. Prune & Merge (velocity-gated for crossing protection)
        7. Extract state estimates

    Args:
        motion_model: State transition / observation model.
        cop_estimator: COP or T-COP DOA estimator instance.
        survival_prob: Target survival probability (p_S).
        detection_prob: Target detection probability (p_D).
        clutter_rate: Expected number of clutter measurements per scan.
        fov_range: Field of view range in radians.
        birth_weight: Weight of each COP-born component.
        prune_threshold: Weight threshold for pruning.
        merge_threshold: Mahalanobis distance threshold for merging.
        max_components: Maximum GM components after pruning.
        birth_pos_std_deg: Birth position uncertainty (degrees).
        birth_vel_std_deg: Birth velocity uncertainty (degrees/scan).
        association_gate_deg: Gating distance for measurement association (degrees).
    """

    def __init__(self, motion_model, cop_estimator,
                 survival_prob=0.95, detection_prob=0.90,
                 clutter_rate=2.0, fov_range=(-np.pi / 2, np.pi / 2),
                 birth_weight=0.1, prune_threshold=1e-5,
                 merge_threshold=4.0, max_components=100,
                 birth_pos_std_deg=2.0, birth_vel_std_deg=5.0,
                 association_gate_deg=8.0):

        self.model = motion_model
        self.cop_estimator = cop_estimator
        self.p_s = survival_prob
        self.p_d = detection_prob

        # Clutter intensity (uniform over FOV)
        fov_size = fov_range[1] - fov_range[0]
        self.clutter_intensity = clutter_rate / fov_size

        self.birth_weight = birth_weight
        self.prune_threshold = prune_threshold
        self.merge_threshold = merge_threshold
        self.max_components = max_components

        # Birth covariance parameters (degrees)
        self.birth_pos_std = np.radians(birth_pos_std_deg)
        self.birth_vel_std = np.radians(birth_vel_std_deg)

        # Association gate
        self.association_gate = np.radians(association_gate_deg)

        # GM-PHD state: list of GaussianComponents
        self.gm_components = []

        # Scan counter and history
        self.scan_count = 0
        self.history = []

        # Track labeling
        self._next_label = 0

    def process_scan(self, X, scan_angles=None):
        """Process one radar scan through the COP-RFS pipeline.

        Novel pipeline:
            1. COP estimation → DOA measurements + spectrum
            2. Predict existing tracks (physics/inertia)
            3. Associate measurements to predictions (Hungarian)
            4. Update confirmed tracks with matched measurements
            5. Birth from unassociated measurements
            6. Prune & Merge (velocity-gated)
            7. Extract estimates

        Args:
            X: Received signal matrix, shape (M, T).
            scan_angles: Angular scan grid (optional).

        Returns:
            estimates: List of (state, covariance, weight) tuples.
            doa_measurements: DOA estimates from COP.
            spectrum: COP spatial spectrum.
        """
        self.scan_count += 1

        if scan_angles is None:
            scan_angles = np.linspace(-np.pi / 2, np.pi / 2, 1801)

        # Step 1: COP DOA estimation
        doa_measurements, spectrum = self.cop_estimator.estimate(X, scan_angles)

        # Step 2: Predict existing tracks (physics/inertia)
        predicted = self._predict_existing()

        # Step 3: Associate measurements to predicted tracks
        # Uses predicted DOAs (from inertia) for identification
        associations, unassoc_meas_idx = self._associate_measurements(
            predicted, doa_measurements)

        # Step 4: Birth from unassociated measurements
        # Created BEFORE update so they receive PHD update in same scan
        unassoc_doas = [doa_measurements[j] for j in unassoc_meas_idx]
        birth_components = self._cop_spectrum_birth(
            spectrum, scan_angles, unassoc_doas)

        # Step 5: Update — confirmed predicted tracks get matched-only update,
        # birth/tentative components get standard PHD update (same scan)
        all_components = predicted + birth_components
        n_predicted = len(predicted)  # Births start at index n_predicted
        updated = self._update_associated(
            all_components, doa_measurements, associations,
            n_predicted=n_predicted)

        # Step 6: Prune and merge (velocity-gated)
        self.gm_components = self._prune_and_merge(updated)

        # Step 7: Extract state estimates
        estimates = self._extract_states()

        # Provide feedback to T-COP if applicable
        self._feedback_to_cop(estimates)

        # Log
        self.history.append({
            'scan': self.scan_count,
            'n_components': len(self.gm_components),
            'n_targets': len(estimates),
            'total_weight': sum(c.weight for c in self.gm_components),
            'n_measurements': len(doa_measurements),
            'n_associated': len(associations),
            'n_births': len(birth_components),
        })

        return estimates, doa_measurements, spectrum

    def _predict_existing(self):
        """Predict existing GM components forward using physics (inertia).

        For each surviving component:
            w_pred = p_s * w
            m_pred = F * m  (constant velocity: position += velocity * dt)
            P_pred = F * P * F^T + Q

        Returns:
            List of predicted GaussianComponents.
        """
        predicted = []

        for comp in self.gm_components:
            w_pred = self.p_s * comp.weight

            if w_pred < self.prune_threshold * 0.1:
                continue

            # State prediction (physics: position += velocity * dt)
            if hasattr(self.model, 'f'):
                m_pred = self.model.f(comp.mean)
            else:
                F = self.model.F(comp.mean)
                m_pred = F @ comp.mean

            # Covariance prediction
            F = self.model.F(comp.mean)
            Q = self.model.Q(comp.mean)
            P_pred = F @ comp.covariance @ F.T + Q

            predicted.append(GaussianComponent(
                w_pred, m_pred, P_pred, comp.label))

        return predicted

    def _associate_measurements(self, predicted, doa_measurements):
        """Associate COP measurements to predicted track positions.

        Uses the physics-based predicted positions (from inertia/velocity)
        to match each measurement to the most likely track. This is the
        'identification before autoregressive update' step.

        For target crossings: even though positions overlap, the PREDICTED
        positions (using learned velocity) will differ, enabling correct
        association.

        Args:
            predicted: List of predicted GaussianComponents.
            doa_measurements: Array of COP DOA estimates (radians).

        Returns:
            associations: List of (track_idx, meas_idx, distance) tuples.
            unassoc_meas_idx: List of unassociated measurement indices.
        """
        # Only associate with confirmed tracks (weight >= 0.3)
        confirmed = [(i, comp) for i, comp in enumerate(predicted)
                     if comp.weight >= 0.3]

        if len(confirmed) == 0 or len(doa_measurements) == 0:
            return [], list(range(len(doa_measurements)))

        n_tracks = len(confirmed)
        n_meas = len(doa_measurements)

        # Cost matrix: angular distance between predicted DOA and measurement
        cost = np.full((n_tracks, n_meas), 1e6)

        for i, (ci, comp) in enumerate(confirmed):
            pred_doa = comp.mean[0]  # Predicted azimuth (already propagated)
            for j, meas_doa in enumerate(doa_measurements):
                d = abs(pred_doa - meas_doa)
                if d < self.association_gate:
                    cost[i, j] = d

        # Hungarian algorithm for optimal assignment
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost)

        associations = []
        matched_tracks = set()
        matched_meas = set()

        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < self.association_gate:
                track_idx = confirmed[r][0]  # Index in predicted list
                associations.append((track_idx, c, cost[r, c]))
                matched_tracks.add(track_idx)
                matched_meas.add(c)

        unassoc_meas_idx = [j for j in range(n_meas) if j not in matched_meas]

        return associations, unassoc_meas_idx

    def _update_associated(self, all_components, doa_measurements, associations,
                           n_predicted=None):
        """Update components using associated measurements.

        For confirmed PREDICTED tracks with an associated measurement:
            Kalman update with matched measurement only (physics-based ID)
            Weight via PHD detection formula

        For confirmed predicted tracks WITHOUT association:
            Missed detection: weight *= (1 - p_d), coast on inertia

        For birth/tentative components:
            Standard PHD update with ALL measurements (same-scan confirmation)

        Args:
            all_components: predicted + birth components.
            doa_measurements: COP DOA estimates.
            associations: (track_idx, meas_idx, dist) tuples.
            n_predicted: Number of predicted components (rest are births).

        Returns:
            List of updated GaussianComponents.
        """
        updated = []
        if n_predicted is None:
            n_predicted = len(all_components)

        # Build track-to-measurement mapping
        track_to_meas = {}
        for track_idx, meas_idx, dist in associations:
            track_to_meas[track_idx] = meas_idx

        for i, comp in enumerate(all_components):
            # Birth components (appended after predicted) always get
            # tentative PHD update, even if their weight is high.
            # This allows same-scan confirmation.
            is_predicted_track = (i < n_predicted)
            is_confirmed = comp.weight >= 0.3 and is_predicted_track

            if is_confirmed:
                # === Confirmed track: explicit association ===
                if i in track_to_meas:
                    # Associated: Kalman update with matched measurement
                    j = track_to_meas[i]
                    z = np.array([doa_measurements[j], 0.0])

                    H = self.model.H(comp.mean)
                    z_pred = H @ comp.mean
                    innov = z - z_pred
                    innov[0] = (innov[0] + np.pi) % (2 * np.pi) - np.pi

                    R = self.model.R()
                    S = H @ comp.covariance @ H.T + R

                    try:
                        S_inv = np.linalg.inv(S)
                    except np.linalg.LinAlgError:
                        S_inv = np.linalg.inv(S + 1e-6 * np.eye(S.shape[0]))

                    K = comp.covariance @ H.T @ S_inv

                    m_upd = comp.mean + K @ innov
                    I_KH = np.eye(len(comp.mean)) - K @ H
                    P_upd = I_KH @ comp.covariance @ I_KH.T + K @ R @ K.T

                    # PHD detection weight update (proper formula):
                    # w = p_d * w_pred * q / (kappa + p_d * w_pred * q)
                    # For well-matched measurement (small innovation), q is large
                    # → w approaches 1.0 quickly (fast birth confirmation)
                    det_S = max(np.linalg.det(S), 1e-30)
                    q = np.exp(-0.5 * innov @ S_inv @ innov) / \
                        np.sqrt((2 * np.pi) ** len(z) * det_S)
                    w_upd = self.p_d * comp.weight * q
                    normalizer = self.clutter_intensity + w_upd
                    w_upd = w_upd / normalizer if normalizer > 0 else comp.weight
                    w_upd = min(w_upd, 1.5)  # Cap

                    updated.append(GaussianComponent(
                        w_upd, m_upd, P_upd, comp.label))
                else:
                    # Not associated: missed detection
                    # Coast on prediction (inertia), reduce weight
                    w_missed = (1 - self.p_d) * comp.weight
                    updated.append(GaussianComponent(
                        w_missed, comp.mean.copy(), comp.covariance.copy(),
                        comp.label))
            else:
                # === Tentative component: standard PHD update ===
                # Missed detection component
                updated.append(GaussianComponent(
                    (1 - self.p_d) * comp.weight,
                    comp.mean.copy(), comp.covariance.copy(), comp.label))

                # Detection update with ALL measurements
                for doa in doa_measurements:
                    z = np.array([doa, 0.0])

                    H = self.model.H(comp.mean)
                    z_pred = H @ comp.mean
                    innov = z - z_pred
                    innov[0] = (innov[0] + np.pi) % (2 * np.pi) - np.pi

                    R = self.model.R()
                    S = H @ comp.covariance @ H.T + R

                    try:
                        S_inv = np.linalg.inv(S)
                    except np.linalg.LinAlgError:
                        S_inv = np.linalg.inv(S + 1e-6 * np.eye(S.shape[0]))

                    K = comp.covariance @ H.T @ S_inv
                    m_upd = comp.mean + K @ innov
                    I_KH = np.eye(len(comp.mean)) - K @ H
                    P_upd = I_KH @ comp.covariance @ I_KH.T + K @ R @ K.T

                    # Gaussian likelihood
                    det_S = max(np.linalg.det(S), 1e-30)
                    q = np.exp(-0.5 * innov @ S_inv @ innov) / \
                        np.sqrt((2 * np.pi) ** len(z) * det_S)

                    w_upd = self.p_d * comp.weight * q
                    normalizer = self.clutter_intensity + w_upd
                    if normalizer > 0:
                        w_upd /= normalizer

                    if w_upd > self.prune_threshold:
                        updated.append(GaussianComponent(
                            w_upd, m_upd, P_upd, comp.label))

        return updated

    def _cop_spectrum_birth(self, spectrum, scan_angles, unassoc_doas):
        """Generate birth components from unassociated COP measurements.

        Only measurements NOT matched to existing tracks create births.
        This is driven by the association step — if a measurement was
        matched to a predicted track position, it's already handled.

        Args:
            spectrum: COP spatial spectrum (normalized to [0, 1]).
            scan_angles: Angular grid.
            unassoc_doas: DOA measurements not associated to any track.

        Returns:
            List of GaussianComponents for birth.
        """
        birth = []

        for doa in unassoc_doas:
            # Find spectrum value at this DOA
            idx = np.argmin(np.abs(scan_angles - doa))
            spec_val = spectrum[idx] if idx < len(spectrum) else 0.5

            # Weight proportional to spectrum height
            w = self.birth_weight * spec_val

            # State: [theta_az, theta_el, theta_dot_az, theta_dot_el]
            dim = self.model.dim_state
            mean = np.zeros(dim)
            mean[0] = doa  # azimuth
            mean[1] = 0.0  # elevation (ULA: unknown)

            # Initial covariance: tight in angle, wide in velocity
            P = np.eye(dim)
            P[0, 0] = self.birth_pos_std ** 2
            P[1, 1] = np.radians(5.0) ** 2
            if dim > 2:
                P[2, 2] = self.birth_vel_std ** 2
            if dim > 3:
                P[3, 3] = self.birth_vel_std ** 2

            label = self._next_label
            self._next_label += 1

            birth.append(GaussianComponent(w, mean, P, label))

        return birth

    def _prune_and_merge(self, components):
        """Prune low-weight and merge close components.

        1. Prune: remove components with w < threshold
        2. Merge: combine components within Mahalanobis distance
           - Velocity-gated: components with significantly different
             velocities are NOT merged even if positions are close.
             This prevents track loss at target crossings.
        3. Cap: keep top max_components by weight
        """
        # Prune
        pruned = [c for c in components if c.weight >= self.prune_threshold]

        if len(pruned) == 0:
            return []

        # Sort by weight (descending) for greedy merge
        pruned.sort(key=lambda c: c.weight, reverse=True)

        # Merge
        merged = []
        used = [False] * len(pruned)

        for i in range(len(pruned)):
            if used[i]:
                continue

            merge_set = [i]
            used[i] = True

            for j in range(i + 1, len(pruned)):
                if used[j]:
                    continue

                # Velocity-gated merge protection:
                # Targets crossing have similar positions but DIFFERENT
                # velocities — merging them destroys one track.
                dim = len(pruned[i].mean)
                if dim >= 4 and pruned[i].weight >= 0.3 and pruned[j].weight >= 0.3:
                    vel_diff = abs(pruned[i].mean[2] - pruned[j].mean[2])
                    if vel_diff > np.radians(2.0):
                        continue

                # Mahalanobis distance
                diff = pruned[j].mean - pruned[i].mean
                try:
                    P_inv = np.linalg.inv(pruned[i].covariance)
                    d2 = diff @ P_inv @ diff
                except np.linalg.LinAlgError:
                    d2 = float('inf')

                if d2 < self.merge_threshold:
                    merge_set.append(j)
                    used[j] = True

            # Merge the set
            w_total = sum(pruned[k].weight for k in merge_set)
            m_merged = sum(pruned[k].weight * pruned[k].mean
                          for k in merge_set) / w_total
            P_merged = np.zeros_like(pruned[i].covariance)
            for k in merge_set:
                diff = pruned[k].mean - m_merged
                P_merged += pruned[k].weight * (
                    pruned[k].covariance + np.outer(diff, diff))
            P_merged /= w_total

            label = pruned[merge_set[0]].label
            merged.append(GaussianComponent(w_total, m_merged, P_merged, label))

        if len(merged) > self.max_components:
            merged.sort(key=lambda c: c.weight, reverse=True)
            merged = merged[:self.max_components]

        return merged

    def _extract_states(self, threshold=0.5):
        """Extract target state estimates from GM components.

        Components with weight >= threshold are considered targets.
        Post-extraction deduplication removes too-close estimates.

        Returns:
            List of (state, covariance, weight) tuples.
        """
        raw_estimates = []

        for comp in self.gm_components:
            if comp.weight >= threshold:
                n_targets = max(1, int(round(comp.weight)))
                for _ in range(n_targets):
                    raw_estimates.append((
                        comp.mean.copy(),
                        comp.covariance.copy(),
                        comp.weight
                    ))

        if len(raw_estimates) <= 1:
            return raw_estimates

        # Deduplicate: remove estimates within 3 degrees of each other
        dedup_threshold = np.radians(3.0)
        raw_estimates.sort(key=lambda e: e[2], reverse=True)

        estimates = []
        used = [False] * len(raw_estimates)

        for i in range(len(raw_estimates)):
            if used[i]:
                continue
            estimates.append(raw_estimates[i])
            used[i] = True
            for j in range(i + 1, len(raw_estimates)):
                if used[j]:
                    continue
                dist = abs(raw_estimates[i][0][0] - raw_estimates[j][0][0])
                if dist < dedup_threshold:
                    used[j] = True

        return estimates

    def _feedback_to_cop(self, estimates):
        """Feed tracked DOAs back to T-COP for temporal prior."""
        from ..doa.temporal_cop import TemporalCOP

        if not isinstance(self.cop_estimator, TemporalCOP):
            return

        if len(estimates) == 0:
            return

        predicted_doas = np.array([est[0][0] for est in estimates])
        n_confirmed = len(estimates)

        self.cop_estimator.set_tracker_predictions(
            predicted_doas, n_confirmed=n_confirmed)

    def get_target_count(self):
        """Expected number of targets (sum of GM weights)."""
        return sum(c.weight for c in self.gm_components)

    def get_doa_estimates(self):
        """Get current DOA estimates from confirmed tracks."""
        estimates = self._extract_states()
        return np.array([est[0][0] for est in estimates]) if estimates else np.array([])

    def get_track_states(self):
        """Get current track states with labels for track identification.

        Returns dict mapping label -> (state_mean, state_cov, weight).
        """
        tracks = {}
        for comp in self.gm_components:
            if comp.weight >= 0.5:
                if comp.label not in tracks or comp.weight > tracks[comp.label][2]:
                    tracks[comp.label] = (
                        comp.mean.copy(),
                        comp.covariance.copy(),
                        comp.weight
                    )
        return tracks

    def reset(self):
        """Reset filter state."""
        self.gm_components = []
        self.scan_count = 0
        self.history = []
        self._next_label = 0
