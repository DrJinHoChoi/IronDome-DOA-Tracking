"""COP-RFS: Gaussian Mixture PHD Filter with COP Spectrum Birth Intensity.

Novel integration of the 2rho-th order Subspace COP DOA estimator with
the Gaussian Mixture Probability Hypothesis Density (GM-PHD) filter.

Key innovation:
    Instead of using a fixed birth model, the COP spatial spectrum is
    used to generate adaptive birth components. This creates an information-
    theoretic link between the higher-order cumulant DOA estimation and
    the Random Finite Set (RFS) based multi-target tracker.

The GM-PHD filter propagates the first-order moment (PHD/intensity) of
the multi-target posterior, avoiding explicit data association.

Patent-relevant novel contributions:
    - COP spectrum as adaptive birth intensity for PHD filter
    - Higher-order cumulant informed target birth/death
    - Underdetermined multi-target tracking (more targets than sensors)
    - T-COP temporal accumulation integrated with PHD prediction

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
    """COP-RFS: GM-PHD Filter with COP Spectrum Birth Intensity.

    The PHD (Probability Hypothesis Density) is represented as a Gaussian
    mixture. The expected number of targets equals the sum of weights.

    Pipeline per scan:
        1. COP DOA estimation -> spectrum
        2. Birth: spectrum peaks -> new Gaussian birth components
        3. Predict: propagate existing GM components forward
        4. Update: incorporate DOA measurements via PHD update
        5. Prune & Merge: manage mixture complexity
        6. Extract: get state estimates from high-weight components

    Args:
        motion_model: State transition / observation model.
        cop_estimator: COP or T-COP DOA estimator instance.
        survival_prob: Target survival probability (p_S).
        detection_prob: Target detection probability (p_D).
        clutter_rate: Expected number of clutter measurements per scan.
        fov_range: Field of view range in radians (e.g., [-pi/2, pi/2]).
        birth_weight: Weight of each COP-born component.
        prune_threshold: Weight threshold for pruning.
        merge_threshold: Mahalanobis distance threshold for merging.
        max_components: Maximum GM components after pruning.
    """

    def __init__(self, motion_model, cop_estimator,
                 survival_prob=0.95, detection_prob=0.90,
                 clutter_rate=2.0, fov_range=(-np.pi / 2, np.pi / 2),
                 birth_weight=0.1, prune_threshold=1e-5,
                 merge_threshold=4.0, max_components=100,
                 birth_pos_std_deg=2.0, birth_vel_std_deg=5.0):

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

        # GM-PHD state: list of GaussianComponents
        self.gm_components = []

        # Scan counter and history
        self.scan_count = 0
        self.history = []

        # Track labeling
        self._next_label = 0

    def process_scan(self, X, scan_angles=None):
        """Process one radar scan through the COP-RFS pipeline.

        Args:
            X: Received signal matrix, shape (M, T).
            scan_angles: Angular scan grid (optional).

        Returns:
            estimates: List of (state, covariance, weight) tuples for
                       extracted targets.
            doa_measurements: DOA estimates from COP.
            spectrum: COP spatial spectrum.
        """
        self.scan_count += 1

        if scan_angles is None:
            scan_angles = np.linspace(-np.pi / 2, np.pi / 2, 1801)

        # Step 1: COP DOA estimation
        doa_measurements, spectrum = self.cop_estimator.estimate(X, scan_angles)

        # Convert to measurement vectors
        measurements = []
        for doa in doa_measurements:
            measurements.append(np.array([doa, 0.0]))  # [az, el=0]

        # Step 2: Generate birth components from COP spectrum
        birth_components = self._cop_spectrum_birth(
            spectrum, scan_angles, doa_measurements)

        # Step 3: Predict
        predicted = self._predict(birth_components)

        # Step 4: Update with measurements
        updated = self._update(predicted, measurements)

        # Step 5: Prune and merge
        self.gm_components = self._prune_and_merge(updated)

        # Step 6: Extract state estimates
        estimates = self._extract_states()

        # Provide feedback to T-COP if applicable
        self._feedback_to_cop(estimates)

        # Log
        self.history.append({
            'scan': self.scan_count,
            'n_components': len(self.gm_components),
            'n_targets': len(estimates),
            'total_weight': sum(c.weight for c in self.gm_components),
            'n_measurements': len(measurements),
        })

        return estimates, doa_measurements, spectrum

    def _cop_spectrum_birth(self, spectrum, scan_angles, doa_measurements):
        """Generate birth components from COP spatial spectrum (gated).

        Novel contribution: Uses the COP spectrum shape to determine
        birth component weights and positions. Higher spectrum values
        at a DOA indicate stronger evidence for a new target.

        Gated birth: If a measurement DOA is close to an existing
        confirmed component, no birth is created (the existing track
        handles it). Only DOAs far from existing tracks create births.

        This prevents duplicate components and reduces false targets.

        Args:
            spectrum: COP spatial spectrum (normalized to [0, 1]).
            scan_angles: Angular grid.
            doa_measurements: Detected DOA peaks.

        Returns:
            List of GaussianComponents for birth.
        """
        birth = []

        # Gating: find DOAs already covered by existing components
        existing_doas = [c.mean[0] for c in self.gm_components
                         if c.weight >= 0.3]
        gate_threshold = np.radians(5.0)  # 5 deg gate

        for doa in doa_measurements:
            # Check if this DOA is near an existing track (gated birth)
            if len(existing_doas) > 0:
                min_dist = np.min(np.abs(np.array(existing_doas) - doa))
                if min_dist < gate_threshold:
                    continue  # Skip: existing track handles this DOA

            # Find spectrum value at this DOA
            idx = np.argmin(np.abs(scan_angles - doa))
            spec_val = spectrum[idx] if idx < len(spectrum) else 0.5

            # Weight proportional to spectrum height
            # Higher spectrum value = stronger birth evidence
            w = self.birth_weight * spec_val

            # State: [theta_az, theta_el, theta_dot_az, theta_dot_el]
            dim = self.model.dim_state
            mean = np.zeros(dim)
            mean[0] = doa  # azimuth
            mean[1] = 0.0  # elevation (ULA: unknown)

            # Initial covariance: tight in angle, wide in velocity
            # Velocity covariance must cover plausible angular rates
            P = np.eye(dim)
            P[0, 0] = self.birth_pos_std ** 2   # azimuth position
            P[1, 1] = np.radians(5.0) ** 2      # elevation (ULA: unobservable)
            if dim > 2:
                P[2, 2] = self.birth_vel_std ** 2  # azimuth rate
            if dim > 3:
                P[3, 3] = self.birth_vel_std ** 2  # elevation rate

            label = self._next_label
            self._next_label += 1

            birth.append(GaussianComponent(w, mean, P, label))

        return birth

    def _predict(self, birth_components):
        """Prediction step: propagate existing components and add births.

        For each surviving component:
            w_pred = p_s * w
            m_pred = F * m  (or f(m) for nonlinear)
            P_pred = F * P * F^T + Q

        Then append birth components.
        """
        predicted = []

        # Predict surviving components
        for comp in self.gm_components:
            w_pred = self.p_s * comp.weight

            if w_pred < self.prune_threshold * 0.1:
                continue  # Skip negligible components early

            # State prediction
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

        # Add birth components
        predicted.extend(birth_components)

        return predicted

    def _update(self, predicted, measurements):
        """PHD update step.

        For each predicted component, create:
        1. Missed detection component: (1 - p_d) * w
        2. For each measurement: detection update component

        The total weight of detection components is normalized by the
        measurement likelihood + clutter intensity.
        """
        updated = []

        # Missed detection components
        for comp in predicted:
            missed = GaussianComponent(
                (1 - self.p_d) * comp.weight,
                comp.mean.copy(),
                comp.covariance.copy(),
                comp.label
            )
            updated.append(missed)

        # Detection update for each measurement
        for z in measurements:
            detection_components = []
            likelihood_sum = 0.0

            for comp in predicted:
                # Predicted measurement
                if hasattr(self.model, 'h'):
                    z_pred = self.model.h(comp.mean)
                    H = self.model.H(comp.mean)
                else:
                    H = self.model.H(comp.mean)
                    z_pred = H @ comp.mean

                # Innovation
                innov = z - z_pred
                # Wrap angles
                innov[0] = (innov[0] + np.pi) % (2 * np.pi) - np.pi

                # Innovation covariance
                R = self.model.R()
                S = H @ comp.covariance @ H.T + R

                # Kalman gain
                try:
                    S_inv = np.linalg.inv(S)
                except np.linalg.LinAlgError:
                    S_inv = np.linalg.inv(S + 1e-6 * np.eye(S.shape[0]))

                K = comp.covariance @ H.T @ S_inv

                # Updated state
                m_upd = comp.mean + K @ innov

                # Updated covariance (Joseph form)
                I_KH = np.eye(len(comp.mean)) - K @ H
                P_upd = I_KH @ comp.covariance @ I_KH.T + K @ R @ K.T

                # Gaussian likelihood
                det_S = max(np.linalg.det(S), 1e-30)
                q = np.exp(-0.5 * innov @ S_inv @ innov) / \
                    np.sqrt((2 * np.pi) ** len(z) * det_S)

                w_upd = self.p_d * comp.weight * q

                detection_components.append(GaussianComponent(
                    w_upd, m_upd, P_upd, comp.label))
                likelihood_sum += w_upd

            # Normalize detection weights
            normalizer = self.clutter_intensity + likelihood_sum
            if normalizer > 0:
                for dc in detection_components:
                    dc.weight /= normalizer
                    updated.append(dc)

        return updated

    def _prune_and_merge(self, components):
        """Prune low-weight and merge close components.

        1. Prune: remove components with w < threshold
        2. Merge: combine components within Mahalanobis distance
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

            # Find components to merge with component i
            merge_set = [i]
            used[i] = True

            for j in range(i + 1, len(pruned)):
                if used[j]:
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

            # Use label from heaviest component
            label = pruned[merge_set[0]].label

            merged.append(GaussianComponent(w_total, m_merged, P_merged, label))

        # Cap maximum components
        if len(merged) > self.max_components:
            merged.sort(key=lambda c: c.weight, reverse=True)
            merged = merged[:self.max_components]

        return merged

    def _extract_states(self, threshold=0.5):
        """Extract target state estimates from GM components.

        Components with weight >= threshold are considered targets.
        Each component with weight w contributes round(w) targets
        (for w > threshold).

        Post-extraction deduplication removes estimates that are too
        close together (within 3 degrees), keeping the higher-weight one.

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
        # Keep the one with higher weight
        dedup_threshold = np.radians(3.0)
        raw_estimates.sort(key=lambda e: e[2], reverse=True)  # Sort by weight desc

        estimates = []
        used = [False] * len(raw_estimates)

        for i in range(len(raw_estimates)):
            if used[i]:
                continue
            estimates.append(raw_estimates[i])
            used[i] = True
            # Mark nearby lower-weight estimates as used
            for j in range(i + 1, len(raw_estimates)):
                if used[j]:
                    continue
                dist = abs(raw_estimates[i][0][0] - raw_estimates[j][0][0])
                if dist < dedup_threshold:
                    used[j] = True

        return estimates

    def _feedback_to_cop(self, estimates):
        """Feed tracked DOAs back to T-COP for temporal prior.

        Novel contribution: PHD track estimates constrain next scan's
        COP subspace estimation.
        """
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
        This enables per-track visualization and analysis.
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
