"""Combat Environment for Mamba-COP-RL.

High-dimensional observation: raw COP spatial spectrum (181-dim).
Long episodes (200 scans) with 4 combat-realistic scenarios.

This environment is designed to expose the advantage of SSM-based
temporal encoding over MLP:
  - Raw spectrum obs (181D) cannot be hand-summarized — Mamba must
    learn what to remember
  - 200 scans create long-range temporal dependencies
  - Combat scenarios require memory of past events:
      Jamming:    remember pre-jam spectrum to recover quickly
      Stealth:    remember disappeared target trajectory
      Saturation: detect sudden influx pattern early
      Formation:  anticipate break-up from formation history

Author: Jin Ho Choi
"""

import numpy as np

from ..signal_model.array import UniformLinearArray
from ..signal_model.signal_generator import generate_snapshots
from ..doa import TemporalCOP
from ..tracking import COPPHD, ConstantVelocity
from ..eval.metrics import gospa


SCENARIO_TYPES = ['jamming', 'stealth', 'saturation', 'formation']


class CombatTrackEnv:
    """Gym-like combat environment for RL-based track management.

    Observation (dim=183):
        [0:181]  Normalized COP spatial spectrum (181 angles, -90° to +90°)
        [181]    scan_fraction  — episode progress 0 → 1
        [182]    snr_proxy      — spectrum peak/floor ratio (normalized)

    Action (dim=3, continuous [-1, 1]):
        [0] birth_weight:    [0.02, 0.30]
        [1] prune_threshold: [1e-6, 1e-3]  (log scale)
        [2] detection_prob:  [0.70, 0.99]
    """

    OBS_DIM = 183
    ACT_DIM = 3
    N_ANGLES = 181

    def __init__(self, M=8, snr_db=10, T=64, n_scans=200, rho=2,
                 scenario_type=None):
        self.M = M
        self.snr_db = snr_db
        self.T = T
        self.n_scans = n_scans
        self.rho = rho
        self.scenario_type = scenario_type  # None = random each episode

        self.ula = UniformLinearArray(M=M, d=0.5)
        self.scan_angles = np.linspace(-np.pi / 2, np.pi / 2, self.N_ANGLES)

        self._scan_idx = 0
        self._scans = None
        self._tracker = None
        self._last_spectrum = np.zeros(self.N_ANGLES)

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        stype = self.scenario_type or np.random.choice(SCENARIO_TYPES)

        generators = {
            'jamming':    self._jamming_scenario,
            'stealth':    self._stealth_scenario,
            'saturation': self._saturation_scenario,
            'formation':  self._formation_scenario,
        }
        self._scans = generators[stype]()
        self._scan_idx = 0
        self._last_spectrum = np.zeros(self.N_ANGLES)

        tcop = TemporalCOP(self.ula, rho=self.rho, alpha=0.85,
                           prior_weight=0.3, search_width_deg=15.0)
        motion = ConstantVelocity(dt=1.0,
                                  process_noise_std=np.radians(1.0))
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
        birth_weight  = self._map_action(action[0], 0.02, 0.30)
        prune_thresh  = 10 ** self._map_action(action[1], -6, -3)
        det_prob      = self._map_action(action[2], 0.70, 0.99)

        self._tracker.birth_weight    = birth_weight
        self._tracker.prune_threshold = prune_thresh
        self._tracker.p_d             = det_prob

        X, true_doas, meta = self._scans[self._scan_idx]
        estimates, doa_meas, spectrum = self._tracker.process_scan(
            X, scan_angles=self.scan_angles)

        # Update spectrum for observation
        if spectrum is not None and len(spectrum) == self.N_ANGLES:
            self._last_spectrum = np.asarray(spectrum, dtype=np.float32)

        est_doas = (np.array([e[0][0] for e in estimates])
                    if estimates else np.array([]))
        n_tracks = len(est_doas)
        n_true   = len(true_doas)

        if n_true > 0:
            ep = est_doas.reshape(-1, 1) if n_tracks > 0 else np.zeros((0, 1))
            tp = true_doas.reshape(-1, 1)
            gospa_val, decomp = gospa(ep, tp,
                                      c=np.radians(10), p=2, alpha=2)
        else:
            gospa_val = 0.0 if n_tracks == 0 else 0.05 * n_tracks
            decomp = {'localization': 0, 'missed': 0, 'false': 0}

        reward  = -gospa_val
        reward -= 0.01 * abs(n_tracks - n_true)
        n_comp  = len(self._tracker.gm_components)
        reward -= 0.001 * max(0, n_comp - 50)

        self._scan_idx += 1
        done = self._scan_idx >= len(self._scans)

        info = {
            'gospa':    gospa_val,
            'n_true':   n_true,
            'n_est':    n_tracks,
            'scenario': meta,
        }

        obs = self._get_obs() if not done else np.zeros(self.OBS_DIM,
                                                         dtype=np.float32)
        return obs, reward, done, info

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self):
        obs  = np.zeros(self.OBS_DIM, dtype=np.float32)
        spec = self._last_spectrum

        # [0:181] normalized spectrum
        mx = spec.max()
        obs[:self.N_ANGLES] = spec / (mx + 1e-8) if mx > 0 else spec

        # [181] episode progress
        obs[181] = self._scan_idx / max(self.n_scans, 1)

        # [182] SNR proxy: peak / noise_floor ratio
        if mx > 0:
            floor = np.percentile(spec, 10) + 1e-8
            obs[182] = min(mx / (floor * 100.0), 1.0)

        return obs

    # ------------------------------------------------------------------
    # Scenario generators
    # ------------------------------------------------------------------

    def _make_X(self, true_doas, snr_db):
        if len(true_doas) == 0:
            return ((np.random.randn(self.M, self.T) +
                     1j * np.random.randn(self.M, self.T)) / np.sqrt(2))
        X, _, _ = generate_snapshots(self.ula, true_doas, snr_db,
                                     self.T, 'non_stationary')
        return X

    def _jamming_scenario(self):
        """Normal targets + electronic jammer that turns on/off.

        Mamba advantage: remembers pre-jam spectrum → fast recovery
        when jammer turns off.
        """
        n_tgt = np.random.randint(2, 5)
        targets = [(np.radians(np.random.uniform(-55, 55)),
                    np.radians(np.random.uniform(-0.5, 0.5)))
                   for _ in range(n_tgt)]

        j_start = np.random.randint(40, 80)
        j_end   = min(j_start + np.random.randint(30, 70),
                      self.n_scans - 20)
        j_doa   = np.radians(np.random.uniform(-80, 80))

        scans = []
        for t in range(self.n_scans):
            doas = np.array([d + v * t for d, v in targets
                             if abs(d + v * t) < np.radians(85)])

            jamming = (j_start <= t < j_end)
            if jamming:
                # -10 dB effective SNR + strong directional jammer
                X = self._make_X(doas, self.snr_db - 10)
                jsteer = self.ula.steering_vector(j_doa)
                jp = 10 ** (self.snr_db / 10) * 5
                jsig = (np.sqrt(jp / 2) *
                        (np.random.randn(self.T) +
                         1j * np.random.randn(self.T)))
                X += jsteer[:, None] * jsig[None, :]
            else:
                X = self._make_X(doas, self.snr_db)

            scans.append((X, doas,
                          {'type': 'jamming', 'jamming': jamming}))
        return scans

    def _stealth_scenario(self):
        """Targets that go low-observable for 15-40 scans then reappear.

        Mamba advantage: hidden state holds trajectory prediction
        during stealth gap → correct re-association on reappearance.
        """
        n_tgt = np.random.randint(2, 5)
        targets = []
        for _ in range(n_tgt):
            d   = np.radians(np.random.uniform(-55, 55))
            v   = np.radians(np.random.uniform(-0.5, 0.5))
            s0  = np.random.randint(30, 100)
            dur = np.random.randint(15, 40)
            targets.append((d, v, s0, s0 + dur))

        scans = []
        for t in range(self.n_scans):
            visible, stealth = [], []
            for d, v, s0, s1 in targets:
                cur = d + v * t
                if abs(cur) >= np.radians(85):
                    continue
                (stealth if s0 <= t < s1 else visible).append(cur)

            true_doas = np.array(sorted(visible + stealth))
            X = self._make_X(np.array(visible), self.snr_db)

            scans.append((X, true_doas,
                          {'type': 'stealth',
                           'stealth_active': len(stealth) > 0,
                           'n_stealth': len(stealth)}))
        return scans

    def _saturation_scenario(self):
        """Small baseline then sudden influx of 6-12 targets.

        Mamba advantage: detects rising-threat pattern early in
        the saturation sequence → raises birth_weight proactively.
        """
        n_init = np.random.randint(2, 4)
        init_tgts = [(np.radians(np.random.uniform(-55, 55)),
                      np.radians(np.random.uniform(-0.3, 0.3)),
                      0, self.n_scans)
                     for _ in range(n_init)]

        t_atk   = np.random.randint(40, 80)
        n_atk   = np.random.randint(6, 12)
        atk_tgts = []
        for _ in range(n_atk):
            d   = np.radians(np.random.uniform(-80, 80))
            v   = np.radians(np.random.uniform(-1.0, 1.0))
            dur = np.random.randint(40, 100)
            atk_tgts.append((d, v, t_atk, t_atk + dur))

        all_tgts = init_tgts + atk_tgts

        scans = []
        for t in range(self.n_scans):
            doas = []
            for d, v, tb, td in all_tgts:
                if tb <= t < td:
                    cur = d + v * (t - tb)
                    if abs(cur) < np.radians(85):
                        doas.append(cur)
            true_doas = np.array(sorted(doas))
            X = self._make_X(true_doas, self.snr_db)

            scans.append((X, true_doas,
                          {'type': 'saturation',
                           'saturation_active': t >= t_atk,
                           'n_targets': len(true_doas)}))
        return scans

    def _formation_scenario(self):
        """Tight angular formation that suddenly breaks apart.

        Mamba advantage: 60-120 scans of formation history allows
        the SSM to model each target's inertia separately, enabling
        fast re-identification after break-up.
        """
        n_form  = np.random.randint(3, 6)
        c_doa   = np.radians(np.random.uniform(-35, 35))
        c_vel   = np.radians(np.random.uniform(-0.3, 0.3))
        spacing = np.radians(np.random.uniform(2.0, 4.0))
        offsets = np.linspace(-(n_form - 1) / 2,
                              (n_form - 1) / 2, n_form) * spacing
        t_break = np.random.randint(60, 130)
        break_v = [np.radians(np.random.uniform(-2.0, 2.0))
                   for _ in range(n_form)]

        scans = []
        for t in range(self.n_scans):
            doas = []
            for i, off in enumerate(offsets):
                if t < t_break:
                    doa = c_doa + off + c_vel * t
                else:
                    doa_at_break = c_doa + off + c_vel * t_break
                    doa = doa_at_break + break_v[i] * (t - t_break)
                if abs(doa) < np.radians(85):
                    doas.append(doa)
            true_doas = np.array(sorted(doas))
            X = self._make_X(true_doas, self.snr_db)

            scans.append((X, true_doas,
                          {'type': 'formation',
                           'intact': t < t_break,
                           'n_formation': n_form}))
        return scans

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _map_action(a, lo, hi):
        return lo + (a + 1) / 2 * (hi - lo)
