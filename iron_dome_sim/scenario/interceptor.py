"""Interception logic for Iron Dome simulation.

Implements threat assessment, impact point prediction,
and interception decision logic.
"""

import numpy as np


class Interceptor:
    """Iron Dome interception system.

    Determines which threats to engage based on predicted impact points,
    track quality, and protected area definitions.

    Args:
        protected_areas: List of (center, radius) tuples defining protected zones.
        intercept_range: Maximum interception range in meters.
        intercept_prob_base: Base interception success probability.
        max_simultaneous: Maximum simultaneous engagements.
    """

    def __init__(self, protected_areas, intercept_range=15000,
                 intercept_prob_base=0.9, max_simultaneous=10):
        self.protected_areas = protected_areas
        self.intercept_range = intercept_range
        self.intercept_prob_base = intercept_prob_base
        self.max_simultaneous = max_simultaneous
        self.active_engagements = []
        self.intercept_log = []

    def assess_threats(self, tracks, track_states):
        """Assess which tracks pose a threat to protected areas.

        Args:
            tracks: List of confirmed Track objects.
            track_states: Dict {track_id: state_vector}.

        Returns:
            threat_list: List of (track_id, threat_level, predicted_impact)
                         sorted by threat_level (descending).
        """
        threat_list = []

        for track in tracks:
            state = track.filter.x
            predicted_impact = self._predict_impact(state)

            if predicted_impact is None:
                continue

            # Check if impact is in a protected area
            threat_level = 0.0
            for center, radius in self.protected_areas:
                dist = np.linalg.norm(predicted_impact[:2] - np.array(center[:2]))
                if dist < radius:
                    # Threat level inversely proportional to distance
                    threat_level = max(threat_level, 1.0 - dist / radius)

            if threat_level > 0:
                threat_list.append({
                    'track_id': track.id,
                    'threat_level': threat_level,
                    'predicted_impact': predicted_impact,
                    'track_quality': track.hit_count / max(track.total_scans, 1),
                })

        # Sort by threat level (highest first)
        threat_list.sort(key=lambda x: x['threat_level'], reverse=True)
        return threat_list

    def decide_intercept(self, threat_list):
        """Decide which threats to intercept.

        Args:
            threat_list: Sorted list from assess_threats().

        Returns:
            engage_list: List of track_ids to engage.
        """
        available_slots = self.max_simultaneous - len(self.active_engagements)
        engage_list = []

        for threat in threat_list:
            if available_slots <= 0:
                break

            track_id = threat['track_id']

            # Skip if already engaged
            if track_id in self.active_engagements:
                continue

            # Only engage if threat level is significant
            if threat['threat_level'] > 0.2:
                engage_list.append(track_id)
                self.active_engagements.append(track_id)
                available_slots -= 1

        return engage_list

    def execute_intercept(self, track_id, track_quality):
        """Execute interception and determine success.

        Args:
            track_id: ID of track to intercept.
            track_quality: Track quality metric (0-1).

        Returns:
            success: Whether interception was successful.
        """
        # Intercept probability depends on track quality
        prob = self.intercept_prob_base * track_quality
        success = np.random.random() < prob

        self.intercept_log.append({
            'track_id': track_id,
            'success': success,
            'probability': prob,
        })

        if track_id in self.active_engagements:
            self.active_engagements.remove(track_id)

        return success

    def _predict_impact(self, state):
        """Predict impact point from current state.

        Assumes ballistic trajectory (gravity only) for prediction.

        Args:
            state: State vector [px, py, pz, vx, vy, vz] or
                   [theta_az, theta_el, ...].

        Returns:
            impact_point: Predicted ground impact [x, y, 0] or None.
        """
        if len(state) >= 6:
            # Cartesian state
            px, py, pz, vx, vy, vz = state[:6]
        else:
            # DOA-only state — cannot predict impact without range
            return None

        if pz <= 0:
            return np.array([px, py, 0.0])

        # Time to ground impact: pz + vz*t - 0.5*g*t^2 = 0
        g = 9.81
        # Quadratic: -0.5*g*t^2 + vz*t + pz = 0
        a = -0.5 * g
        b = vz
        c = pz

        discriminant = b ** 2 - 4 * a * c
        if discriminant < 0:
            return None

        t1 = (-b + np.sqrt(discriminant)) / (2 * a)
        t2 = (-b - np.sqrt(discriminant)) / (2 * a)

        # Take positive time
        t_impact = max(t1, t2)
        if t_impact <= 0:
            return None

        impact_x = px + vx * t_impact
        impact_y = py + vy * t_impact

        return np.array([impact_x, impact_y, 0.0])

    def get_statistics(self):
        """Get interception statistics."""
        if len(self.intercept_log) == 0:
            return {'total': 0, 'success': 0, 'rate': 0.0}

        total = len(self.intercept_log)
        success = sum(1 for log in self.intercept_log if log['success'])
        return {
            'total': total,
            'success': success,
            'rate': success / total,
        }
