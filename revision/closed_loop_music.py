# -*- coding: utf-8 -*-
"""
폐루프(closed-loop) MUSIC = T-MUSIC.

T-COP의 2차-통계 버전: 공분산 시간 누적 + 추적기 예측 기반 부분공간 정련을
**동일한 사영자 블렌딩 메커니즘**(Theorem 2)으로 수행한다. 이로써
2x2 (프론트엔드 {COP, MUSIC} x 루프 {open, closed}) 공정 비교가 가능하고,
폐루프 이득이 COP 고유인지 아니면 일반 메커니즘인지 분리할 수 있다.

리뷰어 공정성 대응: 폐루프를 COP에만 주지 않고 MUSIC에도 동일 적용.
"""
import numpy as np
from iron_dome_sim.doa.music import MUSIC
from iron_dome_sim.doa.spectrum import find_peaks_doa


class TemporalMUSIC(MUSIC):
    def __init__(self, array, num_sources=None, alpha=0.85, prior_weight=0.3,
                 search_width_deg=15.0):
        super().__init__(array, num_sources)
        self.alpha = alpha
        self.prior_weight = prior_weight
        self.search_width = np.radians(search_width_deg)
        self.name = "T-MUSIC"
        self.R_acc = None
        self.scan_count = 0
        self.predicted_doas = None
        self.predicted_covs = None
        self.n_confirmed_tracks = 0

    # 추적기 피드백 (duck-typed; cop_phd_filter._feedback_to_cop 가 호출)
    def set_tracker_predictions(self, predicted_doas, predicted_covs=None,
                                n_confirmed=0, predicted_vels=None):
        self.predicted_doas = (np.array(predicted_doas)
                               if len(predicted_doas) > 0 else None)
        self.predicted_covs = predicted_covs
        self.predicted_vels = (np.array(predicted_vels)
                               if predicted_vels is not None else None)
        self.n_confirmed_tracks = n_confirmed

    def estimate(self, X, scan_angles=None):
        if scan_angles is None:
            scan_angles = self._default_scan_angles()
        P = self.spectrum(X, scan_angles)
        K = self.num_sources if self.num_sources is not None \
            else self._estimate_num_sources(X)
        K = min(K, self.max_sources)
        doa = self._prior_guided_peaks(P, scan_angles, K)
        self.scan_count += 1
        return doa, P

    def spectrum(self, X, scan_angles):
        M, T = X.shape
        R = X @ X.conj().T / T
        # 공분산 시간 누적 (T-COP 의 cumulant 누적에 대응)
        if self.R_acc is None or self.R_acc.shape != R.shape:
            self.R_acc = R.copy()
        else:
            self.R_acc = self.alpha * self.R_acc + (1 - self.alpha) * R
        Rc = self.R_acc

        ev, U = np.linalg.eigh(Rc)
        idx = np.argsort(ev)[::-1]
        U = U[:, idx]
        K = self.num_sources if self.num_sources is not None \
            else self._estimate_num_sources(X)
        K = min(K, M - 1)
        U_s, U_n = U[:, :K], U[:, K:]

        if self.predicted_doas is not None and self.prior_weight > 0:
            U_s, U_n = self._refine(U_s, U_n, M, K)

        P = np.zeros(len(scan_angles))
        for i, th in enumerate(scan_angles):
            a = self.array.steering_vector(th)
            denom = np.real(np.sum(np.abs(U_n.conj().T @ a) ** 2))
            P[i] = 1.0 / (denom + 1e-15)
        m = np.max(P)
        return P / m if m > 0 else P

    def _refine(self, U_s, U_n, M, K):
        """T-COP 와 동일한 사영자 블렌딩 정련 (Theorem 2)."""
        w = self.prior_weight
        pv = [self.array.steering_vector(d) for d in self.predicted_doas]
        A = np.column_stack(pv)
        Q, _ = np.linalg.qr(A, mode="reduced")
        Ku = min(K, U_s.shape[1], Q.shape[1])
        if Ku < 1:
            return U_s, U_n
        Pd = U_s[:, :Ku] @ U_s[:, :Ku].conj().T
        Pp = Q[:, :Ku] @ Q[:, :Ku].conj().T
        Pb = (1 - w) * Pd + w * Pp
        ev, V = np.linalg.eigh(Pb)
        idx = np.argsort(ev)[::-1]
        V = V[:, idx]
        Kb = min(K, M - 1)
        return V[:, :Kb], V[:, Kb:]

    def _prior_guided_peaks(self, P, scan, K):
        if self.predicted_doas is None or len(self.predicted_doas) == 0:
            return find_peaks_doa(P, scan, K)
        allp = find_peaks_doa(P, scan, K + 5)
        conf, rem = [], list(allp)
        for pd in self.predicted_doas:
            if not rem:
                break
            d = np.abs(np.array(rem) - pd)
            j = int(np.argmin(d))
            if d[j] < self.search_width:
                conf.append(rem.pop(j))
        nrem = K - len(conf)
        if nrem > 0 and rem:
            h = [P[int(np.argmin(np.abs(scan - p)))] for p in rem]
            order = np.argsort(h)[::-1]
            for i in range(min(nrem, len(rem))):
                conf.append(rem[order[i]])
        return np.sort(np.array(conf))

    def reset(self):
        self.R_acc = None
        self.scan_count = 0
        self.predicted_doas = None
        self.predicted_covs = None
        self.n_confirmed_tracks = 0
