# -*- coding: utf-8 -*-
r"""
delta-GLMB (delta Generalized Labeled Multi-Bernoulli) tracker
==============================================================
R2.5 가 명시 요청한 SOTA 라벨드-RFS 추적기.  LMB(주변 베르누이 근사)의
*상위* 정밀 모델로, 다중 전역가설(global hypotheses)을 유지한다.

  Vo & Vo, "Labeled RFS and Multi-Object Conjugate Priors," IEEE TSP 2013.
  Vo, Vo & Hoang, "An Efficient Implementation of the GLMB Filter,"
                  IEEE TSP 2017  (Gibbs 결합 예측-갱신 절단).

표현:  posterior ~ sum_h  w_h  delta_{I_h}(L)  [p_h]^X
  각 가설 h = dict(w=가중치, tracks=[{l,m,P}, ...]);  존재 라벨 = 트랙들.

joint predict-update (가설별):
  각 부모가설의 (생존 트랙) + (측정주도 birth) 를 하나의 할당벡터 gamma 로 묶고
  Gibbs 로 표본추출하여 고가중 자식가설만 남긴다.
    gamma_i in {ABSENT(=사망/미탄생), MISSED(=존재·미검출), 0..m-1(=측정 j 검출)}
    자식 w = 부모 w * prod_i eta_i(gamma_i),  측정은 배타적(exclusive).
merge : 동일 라벨집합 가설 병합(가중치 합).   prune : 상위 max_hyp.
extract: 세력분포 rho(n)=sum_{|I|=n} w 의 MAP n*,  그 중 최대가중 가설의 트랙.

COPPHD 와 동일 인터페이스(process_scan / get_doa_estimates / get_track_states)
라 eval_mc 하니스에 그대로 끼워 비교 가능.
"""
import numpy as np

ABSENT, MISSED = -2, -1


def _cv_FQ(dt, sw2):
    F = np.array([[1.0, dt], [0.0, 1.0]])
    Q = sw2 * np.array([[dt**3 / 3.0, dt**2 / 2.0], [dt**2 / 2.0, dt]])
    return F, Q


class GLMBTracker:
    def __init__(self, motion_model, estimator, p_S=0.98, p_D=0.95,
                 clutter_rate=0.3, r_birth=0.3, meas_std_deg=1.0,
                 birth_pos_std_deg=3.0, birth_vel_std_deg=5.0,
                 assoc_gate_deg=10.0, max_hyp=30, max_parent=15,
                 n_gibbs=80, burn=10, extract_r=0.5, min_age=2, fov_rad=np.pi):
        self.model = motion_model
        self.est = estimator
        self.dt = getattr(motion_model, "dt", 1.0)
        self.sw2 = getattr(motion_model, "process_noise_std", np.radians(0.5)) ** 2
        self.p_S, self.p_D = p_S, p_D
        self.kappa = clutter_rate / fov_rad        # uniform clutter intensity
        self.r_B = r_birth
        self.R = np.radians(meas_std_deg) ** 2
        self.Pb = np.diag([np.radians(birth_pos_std_deg) ** 2,
                           np.radians(birth_vel_std_deg) ** 2])
        self.gate2 = np.radians(assoc_gate_deg) ** 2
        self.max_hyp, self.max_parent = max_hyp, max_parent
        self.n_gibbs, self.burn = n_gibbs, burn
        self.extract_r, self.min_age = extract_r, min_age
        self.hyps = [dict(w=1.0, tracks=[])]       # single empty hypothesis
        self.age = {}                              # label -> consecutive confirmed scans
        self.next_label = 0
        self.scan_count = 0

    # ---- RFS interface (matches COPPHD / LMBTracker) ----
    def process_scan(self, X, scan_angles=None):
        Z, spectrum = self.est.estimate(X, scan_angles)
        Z = np.asarray(Z, float).ravel()
        self._predict_update(Z)
        self.scan_count += 1
        return self.get_doa_estimates(), Z, spectrum

    def _marginals(self):
        """label -> marginal existence prob q(l) = sum_{h ni l} w_h."""
        q = {}
        for h in self.hyps:
            for t in h["tracks"]:
                q[t["l"]] = q.get(t["l"], 0.0) + h["w"]
        return q

    def _best_hyp(self):
        """cardinality-MAP, then max-weight hypothesis (coherent joint estimate)."""
        if not self.hyps:
            return None
        card_w = {}
        for h in self.hyps:
            n = len(h["tracks"])
            card_w[n] = card_w.get(n, 0.0) + h["w"]
        n_star = max(card_w, key=card_w.get)        # MAP cardinality
        cands = [h for h in self.hyps if len(h["tracks"]) == n_star]
        return max(cands, key=lambda h: h["w"]) if cands else None

    def get_doa_estimates(self):
        h = self._best_hyp()
        if h is None:
            return np.array([])
        return np.array([t["m"][0] for t in h["tracks"]
                         if self.age.get(t["l"], 0) >= self.min_age])

    def get_track_states(self):
        h = self._best_hyp()
        if h is None:
            return {}
        q = self._marginals()
        return {t["l"]: (t["m"].copy(), t["P"].copy(), float(q.get(t["l"], 1.0)))
                for t in h["tracks"] if self.age.get(t["l"], 0) >= self.min_age}

    def reset(self):
        self.hyps = [dict(w=1.0, tracks=[])]
        self.age = {}
        self.next_label = 0
        self.scan_count = 0

    # ---- joint predict-update ----
    def _predict_update(self, Z):
        m = len(Z)
        H = np.array([[1.0, 0.0]])
        F, Q = _cv_FQ(self.dt, self.sw2)
        birth_labels = [self.next_label + j for j in range(m)]   # shared this scan
        self.next_label += m

        parents = sorted(self.hyps, key=lambda h: h["w"],
                         reverse=True)[:self.max_parent]
        children = []
        for h in parents:
            # ---- predict surviving tracks (KF) ----
            surv = []
            for t in h["tracks"]:
                surv.append(dict(l=t["l"], m=F @ t["m"], P=F @ t["P"] @ F.T + Q))
            n_s = len(surv)
            births = [dict(l=birth_labels[j], m=np.array([Z[j], 0.0]),
                           P=self.Pb.copy(), src=j) for j in range(m)]
            tracks_all = surv + births
            n_all = len(tracks_all)

            # ---- per-track association costs + KF updates ----
            c_absent = np.zeros(n_all)
            c_missed = np.zeros(n_all)
            det = [dict() for _ in range(n_all)]    # i -> {j: (cost, m_upd, P_upd)}
            for i, t in enumerate(tracks_all):
                zhat = float(H @ t["m"])
                S = float(H @ t["P"] @ H.T) + self.R
                K = (t["P"] @ H.T) / S
                Pu = t["P"] - (K @ H @ t["P"])
                if i >= n_s:                        # birth (measurement-driven)
                    c_absent[i] = 1.0 - self.r_B
                    c_missed[i] = 0.0               # born <=> detected by its meas
                    pres = self.r_B
                    j0 = t["src"]
                    dz = Z[j0] - zhat
                    q = np.exp(-0.5 * dz * dz / S) / np.sqrt(2 * np.pi * S)
                    det[i][j0] = (pres * self.p_D * q / max(self.kappa, 1e-12),
                                  t["m"] + K.ravel() * dz, Pu)
                else:                               # survivor
                    c_absent[i] = 1.0 - self.p_S    # death
                    c_missed[i] = self.p_S * (1.0 - self.p_D)
                    pres = self.p_S
                    for j in range(m):
                        dz = Z[j] - zhat
                        if dz * dz <= self.gate2:
                            q = np.exp(-0.5 * dz * dz / S) / np.sqrt(2 * np.pi * S)
                            det[i][j] = (pres * self.p_D * q / max(self.kappa, 1e-12),
                                         t["m"] + K.ravel() * dz, Pu)

            # ---- Gibbs-sample assignment vectors (truncation) ----
            for gamma in self._gibbs(n_all, m, c_absent, c_missed, det):
                ctracks = []
                w_prod = h["w"]
                for i, t in enumerate(tracks_all):
                    g = gamma[i]
                    if g == ABSENT:
                        w_prod *= c_absent[i]
                    elif g == MISSED:                # present, coasting (no measurement)
                        w_prod *= c_missed[i]
                        ctracks.append(dict(l=t["l"], m=t["m"].copy(),
                                            P=t["P"].copy(), det=False))
                    else:                            # present, detected by meas g
                        cost, m_u, P_u = det[i][g]
                        w_prod *= cost
                        ctracks.append(dict(l=t["l"], m=m_u.copy(),
                                            P=P_u.copy(), det=True))
                if w_prod > 0.0:
                    children.append(dict(w=w_prod, tracks=ctracks))

        self.hyps = self._merge_prune(children)

        # ---- confirmation score (hysteresis M-of-N on DETECTION events) ----
        # hit = label DETECTED (measurement-associated) in the cardinality-MAP
        # hypothesis; coasting (missed) or absent => -1. Clutter is detected only
        # at its single birth scan -> score never reaches min_age. A true track is
        # re-detected repeatedly -> confirmed, and hysteresis (cap min_age+2) lets
        # it coast through occasional front-end misses without being de-confirmed.
        h = self._best_hyp()
        detected = (set(t["l"] for t in h["tracks"] if t.get("det"))
                    if h is not None else set())
        a_cap = self.min_age + 2
        new_age = {}
        for l in set(self.age) | detected:
            a = self.age.get(l, 0)
            a = min(a + 1, a_cap) if l in detected else a - 1
            if a > 0:
                new_age[l] = a
        self.age = new_age

    def _gibbs(self, n_all, m, c_absent, c_missed, det):
        gamma = np.full(n_all, ABSENT, dtype=int)
        used = np.zeros(m, dtype=bool)
        out, seen = [], set()
        for it in range(self.n_gibbs):
            for i in range(n_all):
                if gamma[i] >= 0:
                    used[gamma[i]] = False
                opts = [ABSENT, MISSED]
                wts = [max(c_absent[i], 0.0), max(c_missed[i], 0.0)]
                for j, (cost, _, _) in det[i].items():
                    if not used[j]:
                        opts.append(j)
                        wts.append(max(cost, 0.0))
                wts = np.asarray(wts)
                s = wts.sum()
                ch = ABSENT if s <= 0 else opts[int(np.random.choice(len(opts), p=wts / s))]
                gamma[i] = ch
                if ch >= 0:
                    used[ch] = True
            if it >= self.burn:
                key = tuple(gamma.tolist())
                if key not in seen:
                    seen.add(key)
                    out.append(gamma.copy())
        if not out:
            out.append(gamma.copy())
        return out

    def _merge_prune(self, children):
        if not children:
            return [dict(w=1.0, tracks=[])]
        merged = {}                                  # label-set -> entry
        for h in children:
            if h["w"] <= 0:
                continue
            key = frozenset(t["l"] for t in h["tracks"])
            e = merged.get(key)
            if e is None:
                merged[key] = dict(w=h["w"], best=h["w"], tracks=h["tracks"])
            else:
                e["w"] += h["w"]                     # accumulate hypothesis mass
                if h["w"] > e["best"]:               # keep max-weight representative
                    e["best"] = h["w"]
                    e["tracks"] = h["tracks"]
        out = [dict(w=e["w"], tracks=e["tracks"]) for e in merged.values()]
        if not out:
            return [dict(w=1.0, tracks=[])]
        out.sort(key=lambda h: h["w"], reverse=True)
        out = out[:self.max_hyp]
        W = sum(h["w"] for h in out) or 1.0
        for h in out:
            h["w"] /= W
        return out
