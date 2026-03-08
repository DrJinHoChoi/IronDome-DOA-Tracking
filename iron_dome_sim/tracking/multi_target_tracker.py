"""Integrated multi-target tracker.

Combines DOA estimation, data association, state estimation,
and track management into a complete tracking pipeline.
"""

import numpy as np
from .track_manager import TrackManager
from .association import GNN, JPDA


class MultiTargetTracker:
    """Complete multi-target tracking system.

    Pipeline per scan:
    1. DOA estimation → measurements
    2. Predict all existing tracks
    3. Data association (measurements ↔ tracks)
    4. Update associated tracks
    5. Create new tracks from unassociated measurements
    6. Manage track lifecycle (confirm/delete)

    Args:
        doa_estimator: DOA estimation algorithm (SubspaceCOP, MUSIC, etc.).
        model_factory: Callable creating a new motion model instance.
        filter_type: "ekf", "ukf", or "pf".
        association_type: "gnn" or "jpda".
        gate_threshold: Gating threshold for data association.
    """

    def __init__(self, doa_estimator, model_factory, filter_type="ekf",
                 association_type="gnn", gate_threshold=9.21,
                 confirm_M=3, confirm_N=5, max_miss=5):
        self.doa_estimator = doa_estimator
        self.track_manager = TrackManager(
            model_factory, filter_type,
            confirm_M=confirm_M, confirm_N=confirm_N, max_miss=max_miss
        )

        if association_type == "gnn":
            self.associator = GNN(gate_threshold)
        elif association_type == "jpda":
            self.associator = JPDA(gate_threshold)
        else:
            raise ValueError(f"Unknown association type: {association_type}")

        self.association_type = association_type
        self.scan_count = 0
        self.history = []

    def process_scan(self, X, scan_angles=None):
        """Process one radar scan through the complete pipeline.

        Args:
            X: Received signal matrix, shape (M, T).
            scan_angles: Angular scan grid (optional).

        Returns:
            confirmed_tracks: List of confirmed Track objects.
            doa_estimates: DOA estimates from this scan.
            spectrum: Spatial spectrum (if available).
        """
        self.scan_count += 1

        # Step 1: DOA estimation
        doa_estimates, spectrum = self.doa_estimator.estimate(X, scan_angles)

        if len(doa_estimates) == 0:
            # No detections: mark all tracks as missed
            for track in self.track_manager.tracks:
                track.mark_miss()
            self.track_manager.update_status()
            self.history.append({
                'scan': self.scan_count,
                'doa_estimates': doa_estimates,
                'confirmed_tracks': [],
            })
            return self.track_manager.confirmed_tracks, doa_estimates, spectrum

        # Convert DOA estimates to measurement vectors [theta_az, theta_el]
        measurements = np.column_stack([
            doa_estimates,
            np.zeros_like(doa_estimates),  # elevation = 0 for ULA
        ])

        return self.process_measurements(measurements, doa_estimates, spectrum)

    def process_measurements(self, measurements, doa_estimates=None, spectrum=None):
        """Process pre-computed measurements through tracking pipeline.

        Args:
            measurements: Measurement array, shape (N_meas, dim_obs).
            doa_estimates: Original DOA estimates (for logging).
            spectrum: Spatial spectrum (for logging).

        Returns:
            confirmed_tracks, doa_estimates, spectrum
        """
        if doa_estimates is None:
            doa_estimates = measurements[:, 0] if len(measurements) > 0 else np.array([])

        # Step 2: Predict all tracks
        self.track_manager.predict_all()

        # Step 3: Data association
        tracks = self.track_manager.tracks

        if self.association_type == "gnn":
            associations, unassoc_tracks, unassoc_meas = \
                self.associator.associate(tracks, measurements)

            # Step 4: Update associated tracks
            for track_idx, meas_idx in associations.items():
                tracks[track_idx].update(measurements[meas_idx])

            # Mark unassociated tracks as missed
            for track_idx in unassoc_tracks:
                tracks[track_idx].mark_miss()

        elif self.association_type == "jpda":
            beta, unassoc_meas = self.associator.associate(tracks, measurements)

            # Probability-weighted update
            for i, track in enumerate(tracks):
                if beta[i, -1] < 0.99:  # At least one measurement associated
                    self.associator.weighted_update(track, measurements, beta[i])
                else:
                    track.mark_miss()

        # Step 5: Create new tracks from unassociated measurements
        for meas_idx in unassoc_meas:
            self.track_manager.create_track(measurements[meas_idx])

        # Step 6: Track management
        self.track_manager.update_status()

        # Log
        confirmed = self.track_manager.confirmed_tracks
        self.history.append({
            'scan': self.scan_count,
            'doa_estimates': doa_estimates,
            'n_confirmed': len(confirmed),
            'n_total': len(self.track_manager.tracks),
        })

        return confirmed, doa_estimates, spectrum

    def get_track_states(self):
        """Get current state estimates of all confirmed tracks.

        Returns:
            List of (track_id, state_vector) tuples.
        """
        return [(t.id, t.filter.x.copy()) for t in self.track_manager.confirmed_tracks]

    def get_track_histories(self):
        """Get state histories of all confirmed tracks.

        Returns:
            Dict {track_id: np.array of state_history}.
        """
        return {
            t.id: np.array(t.state_history)
            for t in self.track_manager.confirmed_tracks
        }

    def reset(self):
        """Reset tracker state."""
        self.track_manager.reset()
        self.scan_count = 0
        self.history = []
