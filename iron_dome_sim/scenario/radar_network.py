"""Multi-radar network for distributed DOA estimation and fusion.

Supports multiple radar sites with independent DOA estimation
and cross-bearing (LOB intersection) fusion for position estimation.
"""

import numpy as np
from ..signal_model.array import UniformLinearArray, UniformRectangularArray


class RadarSite:
    """Single radar installation.

    Args:
        position: Radar position [x, y, z] in meters.
        array: Antenna array object (ULA or URA).
        name: Human-readable name.
        max_range: Maximum detection range in meters.
        fov: Field of view in radians (symmetric around boresight).
    """

    def __init__(self, position, array, name="Radar", max_range=50000,
                 fov=np.pi):
        self.position = np.array(position, dtype=float)
        self.array = array
        self.name = name
        self.max_range = max_range
        self.fov = fov

    def can_detect(self, target_pos):
        """Check if target is within detection range and FOV."""
        rel = target_pos - self.position
        dist = np.linalg.norm(rel)
        if dist > self.max_range:
            return False
        # Check FOV (azimuth only for simplicity)
        az = np.arctan2(rel[1], rel[0])
        return abs(az) <= self.fov / 2

    def compute_doa(self, target_pos):
        """Compute true DOA angles from radar to target.

        Returns:
            theta_az: Azimuth angle in radians.
            theta_el: Elevation angle in radians.
            range_m: Range in meters.
        """
        rel = target_pos - self.position
        r = np.linalg.norm(rel)
        theta_az = np.arctan2(rel[1], rel[0])
        theta_el = np.arcsin(rel[2] / (r + 1e-10))
        return theta_az, theta_el, r


class RadarNetwork:
    """Network of multiple radar sites for distributed tracking.

    Supports:
    - Independent DOA estimation per radar
    - Cross-bearing fusion (LOB intersection)
    - Centralized track fusion

    Args:
        sites: List of RadarSite objects.
        snr_db: Base SNR in dB.
    """

    def __init__(self, sites, snr_db=10):
        self.sites = sites
        self.snr_db = snr_db

    def generate_all_snapshots(self, target_positions, T=256, signal_type="missile"):
        """Generate received signals at all radar sites.

        Args:
            target_positions: Array of target positions, shape (K, 3).
            T: Number of snapshots.
            signal_type: Signal type for generation.

        Returns:
            List of dicts per radar:
                {'site': RadarSite, 'X': signal, 'true_doas': angles, ...}
        """
        from ..signal_model.signal_generator import generate_snapshots

        results = []

        for site in self.sites:
            # Find targets visible to this radar
            visible_mask = np.array([site.can_detect(tp) for tp in target_positions])
            visible_targets = target_positions[visible_mask]

            if len(visible_targets) == 0:
                results.append({
                    'site': site,
                    'X': None,
                    'true_az': np.array([]),
                    'true_el': np.array([]),
                    'ranges': np.array([]),
                    'n_targets': 0,
                })
                continue

            # Compute DOA angles
            true_az = []
            true_el = []
            ranges = []
            for tp in visible_targets:
                az, el, r = site.compute_doa(tp)
                true_az.append(az)
                true_el.append(el)
                ranges.append(r)

            true_az = np.array(true_az)
            true_el = np.array(true_el)
            ranges = np.array(ranges)

            # Range-dependent SNR
            ref_range = 10000.0
            snr_adj = self.snr_db - 40 * np.log10(ranges / ref_range)
            avg_snr = np.mean(snr_adj)

            # Generate received signal
            X, _, _ = generate_snapshots(
                site.array, true_az, avg_snr, T, signal_type
            )

            results.append({
                'site': site,
                'X': X,
                'true_az': true_az,
                'true_el': true_el,
                'ranges': ranges,
                'n_targets': len(visible_targets),
            })

        return results

    def cross_bearing_fusion(self, radar_doas):
        """Estimate target positions from multiple radar DOA estimates.

        Uses Line-of-Bearing (LOB) intersection from pairs of radars.

        Args:
            radar_doas: List of (radar_site, doa_angles) tuples.

        Returns:
            estimated_positions: Array of estimated 3D positions.
        """
        positions = []

        # Use pairs of radars for triangulation
        for i in range(len(radar_doas)):
            site_i, doas_i = radar_doas[i]
            for j in range(i + 1, len(radar_doas)):
                site_j, doas_j = radar_doas[j]

                # Match DOA estimates between radars (simple nearest-angle)
                for az_i in doas_i:
                    best_pos = None
                    best_err = float('inf')

                    for az_j in doas_j:
                        pos = self._intersect_lob(
                            site_i.position, az_i,
                            site_j.position, az_j
                        )
                        if pos is not None:
                            # Verify consistency
                            err = self._lob_error(pos, site_i.position, az_i,
                                                  site_j.position, az_j)
                            if err < best_err:
                                best_err = err
                                best_pos = pos

                    if best_pos is not None and best_err < 0.1:
                        positions.append(best_pos)

        if len(positions) == 0:
            return np.empty((0, 3))

        # Remove duplicate positions (cluster nearby estimates)
        return self._cluster_positions(np.array(positions))

    def _intersect_lob(self, pos1, az1, pos2, az2):
        """Find intersection of two Lines of Bearing in 2D.

        Args:
            pos1, pos2: Radar positions.
            az1, az2: Azimuth angles.

        Returns:
            Intersection point [x, y, 0] or None if parallel.
        """
        # Direction vectors
        d1 = np.array([np.cos(az1), np.sin(az1)])
        d2 = np.array([np.cos(az2), np.sin(az2)])

        # Check if parallel
        cross = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(cross) < 1e-10:
            return None

        # Solve: pos1 + t1*d1 = pos2 + t2*d2
        dp = pos2[:2] - pos1[:2]
        t1 = (dp[0] * d2[1] - dp[1] * d2[0]) / cross

        if t1 < 0:  # Behind radar
            return None

        intersection = pos1[:2] + t1 * d1
        return np.array([intersection[0], intersection[1], 0.0])

    def _lob_error(self, pos, pos1, az1, pos2, az2):
        """Compute LOB intersection error (sum of angular mismatches)."""
        az1_est = np.arctan2(pos[1] - pos1[1], pos[0] - pos1[0])
        az2_est = np.arctan2(pos[1] - pos2[1], pos[0] - pos2[0])
        err = abs(az1 - az1_est) + abs(az2 - az2_est)
        return err

    def _cluster_positions(self, positions, threshold=500):
        """Cluster nearby position estimates."""
        if len(positions) <= 1:
            return positions

        clustered = [positions[0]]
        for pos in positions[1:]:
            dists = np.linalg.norm(np.array(clustered) - pos, axis=1)
            if np.min(dists) > threshold:
                clustered.append(pos)
            else:
                # Average with nearest cluster
                idx = np.argmin(dists)
                clustered[idx] = (clustered[idx] + pos) / 2

        return np.array(clustered)
