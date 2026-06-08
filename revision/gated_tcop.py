# -*- coding: utf-8 -*-
r"""
게이트 T-COP (GatedTemporalCOP) — v4 (속도 연동 게이트).

목적: 폐루프가 (i) 정지/저속에서는 full T-COP 수준의 이득(누적+정련)을 얻고,
(ii) 빠른 이동에서는 개루프로 폴백해 절대 악화하지 않게 한다.

핵심: 게이트를 추적기의 \emph{속도 추정}(predicted_vels)에 연동.
  motion = mean(|predicted_vels|)  [rad/scan, Kalman 평활값이라 raw jitter보다 강건]
  * motion < vel_thresh (정지/저속):  cumulant 시간누적 ON + Theorem 2 prior 정련 ON
  * motion >= vel_thresh (빠른 이동):  누적 리셋(현재 스캔만) + prior OFF (개루프)
추가로 정지일 때도 prior-데이터 부분공간 불일치가 크면(편향) prior 생략(이중 안전).

이로써 GCL = full CL (정지) / open-loop (이동) 을 자동 선택 -> 어느 영역에서도 손해 없음.
"""
import numpy as np
from iron_dome_sim.doa.temporal_cop import TemporalCOP
from iron_dome_sim.doa.spectrum import find_peaks_doa
from iron_dome_sim.signal_model.cumulant import compute_cumulant_matrix


def _subspace_d2(Ua, Ub, K):
    Ku = min(K, Ua.shape[1], Ub.shape[1])
    if Ku < 1:
        return 1.0
    Pa = Ua[:, :Ku] @ Ua[:, :Ku].conj().T
    Pb = Ub[:, :Ku] @ Ub[:, :Ku].conj().T
    return float(0.5 * np.linalg.norm(Pa - Pb, "fro") ** 2 / Ku)


class GatedTemporalCOP(TemporalCOP):
    def __init__(self, array, rho=2, alpha=0.85, prior_weight=0.5,
                 num_sources=None, gate_delta2=0.25,
                 vel_thresh_deg=0.4, search_width_deg=15.0):
        super().__init__(array, rho=rho, alpha=alpha, prior_weight=prior_weight,
                         search_width_deg=search_width_deg, num_sources=num_sources)
        self.gate_delta2 = gate_delta2
        self.vel_thresh = np.radians(vel_thresh_deg)
        self.name = f"GT-COP-{2*rho}th"
        self.gate_log = []   # (slow, prior_ok) 진단

    def _sig(self, C, K):
        ev, U = np.linalg.eigh(C)
        idx = np.argsort(np.abs(ev))[::-1]
        return U[:, idx][:, :max(1, min(K, C.shape[0] - 1))]

    def _prior_basis(self, M_v):
        pv = []
        for doa in self.predicted_doas:
            a = self.array.virtual_steering_vector(doa, self.rho)
            if len(a) != M_v:
                a = a[:M_v] if len(a) > M_v else np.pad(a, (0, M_v - len(a)))
            pv.append(a)
        Q, _ = np.linalg.qr(np.column_stack(pv), mode="reduced")
        return Q

    def estimate(self, X, scan_angles=None):
        if scan_angles is None:
            scan_angles = self._default_scan_angles()

        C_new = compute_cumulant_matrix(X, self.rho)
        M_v = C_new.shape[0]
        K = self._determine_num_sources(C_new)

        # --- 속도 게이트: 정지/저속인가? ---
        # 보수적 기본값: 속도 정보가 없으면(콜드스타트/미확정) '이동'으로 간주해 개루프.
        # 이렇게 해야 이동 시나리오 초기에 누적이 켜져 표적을 번지게 하는 악순환을 막는다.
        have_vel = self.predicted_vels is not None and len(self.predicted_vels) > 0
        motion = float(np.mean(np.abs(self.predicted_vels))) if have_vel else np.inf
        slow = have_vel and (motion < self.vel_thresh)

        # --- 누적: 정지면 ON, 이동이면 리셋(현재 스캔만) ---
        if slow:
            C_use = self._accumulate_cumulant(C_new)
        else:
            self.C_accumulated = C_new.copy()
            C_use = C_new

        # --- prior 정련: 정지 AND prior 비편향일 때만 ---
        prior_ok = False
        if slow and self.predicted_doas is not None and len(self.predicted_doas) > 0:
            d2 = _subspace_d2(self._sig(C_use, K), self._prior_basis(M_v), K)
            prior_ok = bool(d2 <= self.gate_delta2)
        self.gate_log.append((slow, prior_ok))

        if prior_ok:
            P = self._compute_constrained_spectrum(C_use, scan_angles, K)
            doa = self._prior_guided_peak_detection(P, scan_angles, K)
        else:
            saved = self.predicted_doas
            self.predicted_doas = None          # 개루프 폴백
            P = self._compute_constrained_spectrum(C_use, scan_angles, K)
            self.predicted_doas = saved
            doa = find_peaks_doa(P, scan_angles, K)

        self.scan_count += 1
        return doa, P


class MotionCompTCOP(TemporalCOP):
    r"""운동보상 prior 정련 (누적 없음).

    악순환 회피: 누적을 쓰지 않으므로(번짐 없음) 이동에서도 안전하고, prior 로는
    추적기의 \emph{예측}(F m = 현재+속도*dt, cop_phd_filter 가 전달)을 받는다.
    예측이 정확하면(CV 성립) prior 가 현재 위치를 맞춰 편향이 사라져, Theorem 2 의
    정련이 \emph{이동 표적까지} MSE 를 낮춘다. 부분공간 불일치(d2)가 크면(기동 등
    예측 실패) prior 를 버리고 개루프로 폴백.
    """
    def __init__(self, array, rho=2, prior_weight=0.5, num_sources=None,
                 gate_delta2=0.3, search_width_deg=15.0):
        super().__init__(array, rho=rho, alpha=1.0, prior_weight=prior_weight,
                         search_width_deg=search_width_deg, num_sources=num_sources)
        self.gate_delta2 = gate_delta2
        self.name = f"MC-COP-{2*rho}th"

    def estimate(self, X, scan_angles=None):
        if scan_angles is None:
            scan_angles = self._default_scan_angles()
        C_use = compute_cumulant_matrix(X, self.rho)   # 누적 없음
        self.C_accumulated = C_use
        M_v = C_use.shape[0]
        K = self._determine_num_sources(C_use)

        prior_ok = False
        if self.predicted_doas is not None and len(self.predicted_doas) > 0:
            U = self._sig_mc(C_use, K)
            Q = self._prior_basis_mc(M_v)
            d2 = _subspace_d2(U, Q, K)
            prior_ok = bool(d2 <= self.gate_delta2)

        if self.predicted_doas is not None and len(self.predicted_doas) > 0:
            if prior_ok:                       # 저편향 prior: 부분공간도 정련
                P = self._compute_constrained_spectrum(C_use, scan_angles, K)
            else:                              # 고편향 prior: 부분공간 정련은 생략
                saved = self.predicted_doas
                self.predicted_doas = None
                P = self._compute_constrained_spectrum(C_use, scan_angles, K)
                self.predicted_doas = saved
            # 피크 검출은 항상 예측 위치를 활용(저위험 해상도 보조)
            doa = self._prior_guided_peak_detection(P, scan_angles, K)
        else:
            P = self._compute_constrained_spectrum(C_use, scan_angles, K)
            doa = find_peaks_doa(P, scan_angles, K)
        self.scan_count += 1
        return doa, P

    def _sig_mc(self, C, K):
        ev, U = np.linalg.eigh(C)
        idx = np.argsort(np.abs(ev))[::-1]
        return U[:, idx][:, :max(1, min(K, C.shape[0] - 1))]

    def _prior_basis_mc(self, M_v):
        pv = []
        for doa in self.predicted_doas:
            a = self.array.virtual_steering_vector(doa, self.rho)
            if len(a) != M_v:
                a = a[:M_v] if len(a) > M_v else np.pad(a, (0, M_v - len(a)))
            pv.append(a)
        Q, _ = np.linalg.qr(np.column_stack(pv), mode="reduced")
        return Q
