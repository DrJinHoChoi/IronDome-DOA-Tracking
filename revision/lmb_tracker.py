# -*- coding: utf-8 -*-
r"""
Labeled Multi-Bernoulli (LMB) tracker — identity-aware RFS baseline.

리뷰어(R2.5/R1.4)가 요구한 SOTA 정체성-인식 추적 비교용. Vo--Vo--Phung /
Reuter 의 LMB 를 가우시안 + 측정주도 birth + Gibbs 결합 데이터연관으로 구현.

각 트랙 = (label, r: 존재확률, m: [theta,thetadot], P).
  predict: 생존 r<-p_S r, 칼만 예측.
  update : Gibbs 로 배타적(exclusive) 결합연관의 주변확률 beta_ij 추정 ->
           트랙별 베르누이 존재·상태 갱신 (JIPDA 형). 미연관 측정 -> birth.
  extract: r>=0.5 트랙.

COPPHD 와 동일 인터페이스(process_scan / get_doa_estimates / get_track_states)
라 eval_mc 하니스에 그대로 끼워 비교 가능.
"""
import numpy as np


def _cv_FQ(dt, sw2):
    F = np.array([[1.0, dt], [0.0, 1.0]])
    Q = sw2 * np.array([[dt**3 / 3.0, dt**2 / 2.0], [dt**2 / 2.0, dt]])
    return F, Q


def _ca_FQ(dt, sw2):
    """Physics-based constant-acceleration model: [theta, theta_dot, theta_ddot]."""
    F = np.array([[1.0, dt, 0.5 * dt * dt],
                  [0.0, 1.0, dt],
                  [0.0, 0.0, 1.0]])
    Q = sw2 * np.array([[dt**5 / 20.0, dt**4 / 8.0, dt**3 / 6.0],
                        [dt**4 / 8.0,  dt**3 / 3.0, dt**2 / 2.0],
                        [dt**3 / 6.0,  dt**2 / 2.0, dt]])
    return F, Q


class LMBTracker:
    def __init__(self, motion_model, estimator, p_S=0.95, p_D=0.95,
                 clutter_rate=0.3, r_birth=0.3, meas_std_deg=1.0,
                 birth_pos_std_deg=3.0, birth_vel_std_deg=5.0,
                 birth_acc_std_deg=2.0, motion='cv',
                 assoc_gate_deg=10.0, prune_r=1e-3, extract_r=0.5,
                 max_tracks=40, n_gibbs=120, burn=20, fov_rad=np.pi):
        self.model = motion_model
        self.est = estimator
        self.dt = getattr(motion_model, "dt", 1.0)
        self.sw2 = getattr(motion_model, "process_noise_std", np.radians(0.5)) ** 2
        self.motion = motion
        if motion == 'ca':                           # physics: constant-acceleration
            self.F, self.Q = _ca_FQ(self.dt, self.sw2)
            self.H = np.array([[1.0, 0.0, 0.0]])
            self.Pb = np.diag([np.radians(birth_pos_std_deg) ** 2,
                               np.radians(birth_vel_std_deg) ** 2,
                               np.radians(birth_acc_std_deg) ** 2])
            self.dim = 3
        else:
            self.F, self.Q = _cv_FQ(self.dt, self.sw2)
            self.H = np.array([[1.0, 0.0]])
            self.Pb = np.diag([np.radians(birth_pos_std_deg) ** 2,
                               np.radians(birth_vel_std_deg) ** 2])
            self.dim = 2
        self.p_S, self.p_D = p_S, p_D
        self.kappa = clutter_rate / fov_rad          # clutter intensity (uniform)
        self.r_B = r_birth
        self.R = np.radians(meas_std_deg) ** 2
        self.gate2 = np.radians(assoc_gate_deg) ** 2
        self.prune_r, self.extract_r = prune_r, extract_r
        self.max_tracks, self.n_gibbs, self.burn = max_tracks, n_gibbs, burn
        self.tracks = []          # list of dict(l, r, m(2,), P(2,2))
        self.next_label = 0
        self.scan_count = 0

    # ---- RFS interface (matches COPPHD) ----
    def process_scan(self, X, scan_angles=None):
        self._feed_prior()                     # closed-loop: prediction -> front-end
        Z, spectrum = self.est.estimate(X, scan_angles)
        Z = np.asarray(Z, float).ravel()
        self._predict()
        self._update(Z)
        self.scan_count += 1
        return self.get_doa_estimates(), Z, spectrum

    def _feed_prior(self):
        """Closed-loop coupling: feed confirmed tracks' motion-compensated DOA
        predictions (theta + thetadot*dt) to the front-end if it accepts a
        temporal prior. No-op for an open-loop estimator (duck-typed)."""
        if not hasattr(self.est, "set_tracker_predictions"):
            return
        pdoas, pvels = [], []
        for t in self.tracks:
            if t["r"] >= self.extract_r:
                pd = t["m"][0] + t["m"][1] * self.dt
                if self.dim >= 3:                  # CA: + 1/2 a dt^2
                    pd += 0.5 * t["m"][2] * self.dt * self.dt
                pdoas.append(pd)
                pvels.append(t["m"][1])
        self.est.set_tracker_predictions(np.array(pdoas),
                                         predicted_vels=np.array(pvels))

    def _birth_mean(self, z):
        m = np.zeros(self.dim)
        m[0] = z
        return m

    def get_doa_estimates(self):
        return np.array([t["m"][0] for t in self.tracks if t["r"] >= self.extract_r])

    def get_track_states(self):
        return {t["l"]: (t["m"].copy(), t["P"].copy(), t["r"])
                for t in self.tracks if t["r"] >= self.extract_r}

    def reset(self):
        self.tracks = []; self.next_label = 0; self.scan_count = 0

    # ---- predict ----
    def _predict(self):
        F, Q = self.F, self.Q
        for t in self.tracks:
            t["r"] = self.p_S * t["r"]
            t["m"] = F @ t["m"]
            t["P"] = F @ t["P"] @ F.T + Q

    # ---- update ----
    def _update(self, Z):
        n, m = len(self.tracks), len(Z)
        H = self.H
        if n > 0 and m > 0:
            A = np.zeros((n, m))      # detection weights (rel. to clutter)
            A0 = np.zeros(n)          # "missed or absent" weight
            Kg, mij = [], []
            for i, t in enumerate(self.tracks):
                zhat = float(H @ t["m"]); S = float(H @ t["P"] @ H.T) + self.R
                K = (t["P"] @ H.T) / S
                Kg.append(K)
                mi = []
                for j in range(m):
                    dz = Z[j] - zhat
                    if dz * dz <= self.gate2:               # association gate
                        q = np.exp(-0.5 * dz * dz / S) / np.sqrt(2 * np.pi * S)
                        A[i, j] = self.p_D * t["r"] * q / max(self.kappa, 1e-12)
                    mi.append(t["m"] + (K.ravel() * dz))
                mij.append(mi)
                A0[i] = 1.0 - self.p_D * t["r"]
            beta, beta0 = self._gibbs(A, A0)
            # Bernoulli + state update per track
            Pupd = []
            for i, t in enumerate(self.tracks):
                S = float(H @ t["P"] @ H.T) + self.R
                Pi = t["P"] - (Kg[i] @ H @ t["P"])
                w = beta[i].copy(); w0 = beta0[i]
                tot = w.sum() + w0
                if tot <= 1e-12:
                    t["r"] = t["r"] * (1 - self.p_D); continue
                # existence: detected mass + (missed-but-exists) part of w0
                r_exist_given_miss = (t["r"] * (1 - self.p_D)) / max(1 - t["r"] * self.p_D, 1e-9)
                t["r"] = float(np.clip(w.sum() + w0 * r_exist_given_miss, 1e-4, 1.0 - 1e-6))
                # state: moment-matched mixture (miss keeps predicted m)
                mmix = w0 * t["m"] + sum(w[j] * mij[i][j] for j in range(m))
                mmix = mmix / tot
                t["m"] = mmix
                t["P"] = (w0 * t["P"] + sum(w[j] * Pi for j in range(m))) / tot
            # birth from measurements with little association mass
            assoc_mass = beta.sum(axis=0) if n > 0 else np.zeros(m)
        else:
            assoc_mass = np.zeros(m)

        self._birth(Z, assoc_mass)
        self._prune()

    def _gibbs(self, A, A0):
        n, m = A.shape
        beta = np.zeros((n, m)); beta0 = np.zeros(n)
        gamma = -np.ones(n, dtype=int)          # all start missed
        used = np.zeros(m, dtype=bool)
        cnt = 0
        for it in range(self.n_gibbs):
            for i in range(n):
                if gamma[i] >= 0:
                    used[gamma[i]] = False
                cand = [-1]; wts = [A0[i]]
                for j in range(m):
                    if not used[j] and A[i, j] > 0:
                        cand.append(j); wts.append(A[i, j])
                wts = np.asarray(wts); s = wts.sum()
                if s <= 0:
                    ch = -1
                else:
                    ch = cand[int(np.random.choice(len(cand), p=wts / s))]
                gamma[i] = ch
                if ch >= 0:
                    used[ch] = True
            if it >= self.burn:
                cnt += 1
                for i in range(n):
                    if gamma[i] >= 0:
                        beta[i, gamma[i]] += 1.0
                    else:
                        beta0[i] += 1.0
        if cnt > 0:
            beta /= cnt; beta0 /= cnt
        return beta, beta0

    def _birth(self, Z, assoc_mass):
        for j in range(len(Z)):
            if assoc_mass[j] < 0.5:              # measurement not well explained
                self.tracks.append(dict(
                    l=self.next_label, r=self.r_B * (1 - assoc_mass[j]),
                    m=self._birth_mean(Z[j]), P=self.Pb.copy()))
                self.next_label += 1

    def _prune(self):
        self.tracks = [t for t in self.tracks if t["r"] >= self.prune_r]
        if len(self.tracks) > self.max_tracks:
            self.tracks.sort(key=lambda t: t["r"], reverse=True)
            self.tracks = self.tracks[:self.max_tracks]
