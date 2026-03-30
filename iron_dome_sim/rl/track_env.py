"""RL Environment for Track Management in COP-PHD.

Wraps the COP-PHD tracker as an MDP where the RL agent decides
track birth/keep/delete at each scan, replacing fixed thresholds.

State: per-scan global features + per-track features
Action: birth_weight, prune_threshold, merge_threshold (continuous)
Reward: -GOSPA (lower GOSPA = higher reward)

Author: Jin Ho Choi
"""

import numpy as np
from copy import deepcopy

from ..signal_model.array import UniformLinearArray
from ..signal_model.signal_generator import generate_snapshots
from ..doa import TemporalCOP
from ..tracking import COPPHD, ConstantVelocity
from ..eval.metrics import gospa


class TrackManagementEnv:
    """Gym-like environment for RL-based track management.

    The agent observes tracker state and outputs adaptive parameters
    that replace fixed thresholds in the GM-PHD filter.

    Observation (dim=12):
        [0] n_tracks (normalized)
        [1] n_measurements (normalized)
        [2] measurement_density
        [3] total_weight (sum of GM weights)
        [4] n_components (normalized)
        [5] n_tracks_delta (change from last scan)
        [6] n_meas_delta
        [7] avg_track_weight
        [8] max_track_weight
        [9] min_track_weight
        [10] scan_fraction (progress through episode)
        [11] snr_proxy (spectrum peak/floor ratio)

    Action (dim=3, continuous [-1, 1] → mapped to parameter ranges):
        [0] birth_weight:     [0.02, 0.30]
        [1] prune_threshold:  [1e-6, 1e-3]
        [2] detection_prob:   [0.70, 0.99]
    """

    OBS_DIM = 12
    ACT_DIM = 3

    def __init__(self, M=8, snr_db=5, T=64, n_scans=40, rho=2,
                 scenario_fn=None):
        self.M = M
        self.snr_db = snr_db
        self.T = T
        self.n_scans = n_scans
        self.rho = rho
        self.scenario_fn = scenario_fn or self._default_scenario

        self.ula = UniformLinearArray(M=M, d=0.5)
        self.scan_angles = np.linspace(-np.pi / 2, np.pi / 2, 181)

        # State tracking
        self._scan_idx = 0
        self._scans = None
        self._tracker = None
        self._prev_n_tracks = 0
        self._prev_n_meas = 0
        self._episode_gospa = []

    def reset(self, seed=None):
        """Reset environment for a new episode.

        Returns:
            obs: Initial observation vector.
        """
        if seed is not None:
            np.random.seed(seed)

        self._scans = self.scenario_fn()
        self._scan_idx = 0
        self._prev_n_tracks = 0
        self._prev_n_meas = 0
        self._episode_gospa = []

        # Create fresh tracker with T-COP
        tcop = TemporalCOP(self.ula, rho=self.rho, alpha=0.85,
                           prior_weight=0.3, search_width_deg=15.0)
        motion = ConstantVelocity(dt=1.0, process_noise_std=np.radians(1.0))
        self._tracker = COPPHD(
            motion_model=motion,
            cop_estimator=tcop,
            survival_prob=0.95,
            detection_prob=0.90,
            clutter_rate=2.0,
            birth_weight=0.1,
            prune_threshold=1e-5,
            merge_threshold=4.0,
            max_components=100,
            birth_pos_std_deg=2.0,
            birth_vel_std_deg=5.0,
            association_gate_deg=8.0,
            use_physics=True,
        )

        return self._get_obs()

    def step(self, action):
        """Execute one scan with RL-chosen parameters.

        Args:
            action: np.array of shape (3,), values in [-1, 1].

        Returns:
            obs: Next observation.
            reward: Scalar reward.
            done: Whether episode is finished.
            info: Dict with metrics.
        """
        # Map action [-1, 1] → parameter ranges
        birth_weight = self._map_action(action[0], 0.02, 0.30)
        prune_thresh = 10 ** self._map_action(action[1], -6, -3)  # log scale
        det_prob = self._map_action(action[2], 0.70, 0.99)

        # Apply RL-chosen parameters
        self._tracker.birth_weight = birth_weight
        self._tracker.prune_threshold = prune_thresh
        self._tracker.p_d = det_prob

        # Run one scan
        X, true_doas = self._scans[self._scan_idx]
        estimates, doa_meas, spectrum = self._tracker.process_scan(
            X, scan_angles=self.scan_angles)

        # Compute metrics
        est_doas = np.array([e[0][0] for e in estimates]) if estimates else np.array([])
        n_tracks = len(est_doas)
        n_meas = len(doa_meas)

        if len(true_doas) > 0:
            est_pos = est_doas.reshape(-1, 1) if len(est_doas) > 0 else np.zeros((0, 1))
            true_pos = true_doas.reshape(-1, 1)
            gospa_val, decomp = gospa(est_pos, true_pos,
                                      c=np.radians(10), p=2, alpha=2)
        else:
            gospa_val = 0.0 if n_tracks == 0 else 0.05 * n_tracks
            decomp = {'localization': 0, 'missed': 0, 'false': 0}

        # Reward: -GOSPA with component penalties
        reward = -gospa_val
        # Bonus for correct target count
        n_true = len(true_doas)
        count_error = abs(n_tracks - n_true)
        reward -= 0.01 * count_error
        # Small penalty for excessive components (memory pressure on Edge)
        n_comp = len(self._tracker.gm_components)
        reward -= 0.001 * max(0, n_comp - 50)

        self._episode_gospa.append(gospa_val)

        # Update state
        self._prev_n_tracks = n_tracks
        self._prev_n_meas = n_meas
        self._scan_idx += 1

        done = self._scan_idx >= len(self._scans)

        info = {
            'gospa': gospa_val,
            'gospa_loc': decomp['localization'],
            'gospa_miss': decomp['missed'],
            'gospa_false': decomp['false'],
            'n_true': n_true,
            'n_est': n_tracks,
            'n_meas': n_meas,
            'n_components': n_comp,
            'birth_weight': birth_weight,
            'prune_threshold': prune_thresh,
            'detection_prob': det_prob,
        }

        obs = self._get_obs() if not done else np.zeros(self.OBS_DIM)
        return obs, reward, done, info

    def _get_obs(self):
        """Extract observation from current tracker state."""
        obs = np.zeros(self.OBS_DIM, dtype=np.float32)

        tracker = self._tracker
        components = tracker.gm_components
        n_tracks = sum(1 for c in components if c.weight >= 0.5)
        weights = [c.weight for c in components] if components else [0]

        obs[0] = n_tracks / 15.0  # normalized
        obs[1] = self._prev_n_meas / 15.0
        obs[2] = self._prev_n_meas / np.pi  # density per radian
        obs[3] = sum(weights) / 10.0
        obs[4] = len(components) / 100.0
        obs[5] = (n_tracks - self._prev_n_tracks) / 5.0  # delta
        obs[6] = 0.0  # will be updated after scan
        obs[7] = np.mean(weights) if weights else 0.0
        obs[8] = max(weights) if weights else 0.0
        obs[9] = min(weights) if weights else 0.0
        obs[10] = self._scan_idx / max(self.n_scans, 1)
        # SNR proxy: use average GOSPA as proxy
        obs[11] = np.mean(self._episode_gospa[-5:]) if self._episode_gospa else 0.0

        return obs

    @staticmethod
    def _map_action(a, lo, hi):
        """Map action from [-1, 1] to [lo, hi]."""
        return lo + (a + 1) / 2 * (hi - lo)

    def _default_scenario(self):
        """Generate a random birth-death scenario for training."""
        n_scans = self.n_scans
        ula = self.ula

        # Random number of target groups
        n_groups = np.random.randint(2, 5)
        targets = []

        for _ in range(n_groups):
            birth = np.random.randint(0, n_scans // 2)
            death = np.random.randint(birth + 10, min(birth + n_scans, n_scans))
            doa_start = np.random.uniform(-70, 70)
            velocity = np.random.uniform(-1.0, 1.0)
            targets.append((birth, death, doa_start, velocity))

        scans = []
        for scan in range(n_scans):
            active = []
            for birth, death, doa0, vel in targets:
                if birth <= scan < death:
                    doa = doa0 + vel * (scan - birth)
                    if -85 <= doa <= 85:
                        active.append(doa)

            true_doas = np.radians(np.array(sorted(active)))

            if len(true_doas) == 0:
                X = (np.random.randn(self.M, self.T) +
                     1j * np.random.randn(self.M, self.T)) / np.sqrt(2)
            else:
                X, _, _ = generate_snapshots(
                    ula, true_doas, self.snr_db, self.T, 'non_stationary')

            scans.append((X, true_doas))

        return scans
