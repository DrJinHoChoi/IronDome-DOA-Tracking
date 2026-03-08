"""Track lifecycle management.

Handles track creation (tentative), confirmation (M/N logic),
and deletion based on miss count.
"""

import numpy as np
from .filters import ExtendedKalmanFilter


class Track:
    """Single target track."""

    _next_id = 0

    def __init__(self, filter_obj, measurement):
        self.id = Track._next_id
        Track._next_id += 1
        self.filter = filter_obj
        self.status = "tentative"  # tentative → confirmed → deleted
        self.hit_count = 1
        self.miss_count = 0
        self.total_scans = 1
        self.history = [measurement.copy()]
        self.state_history = [filter_obj.x.copy()]

    def predict(self):
        x_pred, P_pred = self.filter.predict()
        return x_pred, P_pred

    def update(self, measurement):
        x_upd, P_upd = self.filter.update(measurement)
        self.hit_count += 1
        self.miss_count = 0
        self.total_scans += 1
        self.history.append(measurement.copy())
        self.state_history.append(x_upd.copy())

    def mark_miss(self):
        self.miss_count += 1
        self.total_scans += 1
        self.state_history.append(self.filter.x.copy())

    @property
    def is_confirmed(self):
        return self.status == "confirmed"

    @property
    def age(self):
        return self.total_scans


class TrackManager:
    """Manages track lifecycle: creation, confirmation, deletion.

    Args:
        model_factory: Callable that creates a new motion model.
        filter_type: "ekf", "ukf", or "pf".
        confirm_M: Number of hits required for confirmation.
        confirm_N: Window of scans for M/N logic.
        max_miss: Maximum consecutive misses before deletion.
        init_P_scale: Initial covariance scaling factor.
    """

    def __init__(self, model_factory, filter_type="ekf",
                 confirm_M=3, confirm_N=5, max_miss=5, init_P_scale=1.0):
        self.model_factory = model_factory
        self.filter_type = filter_type
        self.confirm_M = confirm_M
        self.confirm_N = confirm_N
        self.max_miss = max_miss
        self.init_P_scale = init_P_scale
        self.tracks = []

    def create_track(self, measurement):
        """Create a new tentative track from an unassociated measurement."""
        model = self.model_factory()

        # Initialize state from measurement
        x0 = np.zeros(model.dim_state)
        x0[:model.dim_obs] = measurement
        P0 = np.eye(model.dim_state) * self.init_P_scale

        if self.filter_type == "ekf":
            filt = ExtendedKalmanFilter(model, x0, P0)
        elif self.filter_type == "ukf":
            from .filters import UnscentedKalmanFilter
            filt = UnscentedKalmanFilter(model, x0, P0)
        elif self.filter_type == "pf":
            from .filters import ParticleFilter
            filt = ParticleFilter(model, x0, P0)
        else:
            raise ValueError(f"Unknown filter type: {self.filter_type}")

        track = Track(filt, measurement)
        self.tracks.append(track)
        return track

    def update_status(self):
        """Update track statuses based on M/N logic and miss count."""
        for track in self.tracks:
            if track.status == "tentative":
                # M/N confirmation logic
                if track.hit_count >= self.confirm_M:
                    track.status = "confirmed"
                elif track.total_scans >= self.confirm_N:
                    # Failed to confirm within N scans → delete
                    track.status = "deleted"

            if track.miss_count >= self.max_miss:
                track.status = "deleted"

        # Remove deleted tracks
        self.tracks = [t for t in self.tracks if t.status != "deleted"]

    @property
    def confirmed_tracks(self):
        return [t for t in self.tracks if t.status == "confirmed"]

    @property
    def all_tracks(self):
        return self.tracks

    def predict_all(self):
        """Predict all tracks forward one step."""
        for track in self.tracks:
            track.predict()

    def reset(self):
        """Clear all tracks."""
        self.tracks = []
        Track._next_id = 0
