# -*- coding: utf-8 -*-
"""
COP-RFS 재투고 — 신뢰 가능한 Monte-Carlo 평가 하니스
=====================================================
기존 단일-시드 그림 스크립트(tests/generate_*.py)를 대체한다.

수정 사항 (리뷰어 대응):
  * R2#4 / R3#9 : ≥500회 독립 시드 MC + 95% 신뢰구간.
  * RMSE 부정확성 : 기존 metrics.rmse_doa 는 누락표적마다 (π/2)²≈90°
    페널티를 더해 검출 개수에 극도로 민감(재현 불가). 여기서는
    **GOSPA-매칭쌍 한정 localization-RMSE** 로 분리 보고.
  * GOSPA 를 localization / missed / false 로 분해해 '왜' 차이가 나는지 표시.
  * switch 는 PHD 라벨 기반(get_track_states)으로 정확히 카운트.

시나리오:
  k10 : underdetermined end-to-end (K=10>M-1=7), COP-RFS vs MUSIC-PHD
  k4  : determined crossing 2x2 ablation (Table I)

사용:
  python revision/eval_mc.py --scenario k10 --trials 8 --T 256 --jobs 4   # 빠른 검증
  python revision/eval_mc.py --scenario k10 --trials 500 --jobs 8         # 최종
"""
from __future__ import annotations
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.optimize import linear_sum_assignment

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # revision/ (closed_loop_music)


# ===================================================================== #
#  시나리오 정의 (원본 tests/generate_*.py 와 동일)                      #
# ===================================================================== #
SCENARIOS = {
    "k10": dict(           # end-to-end underdetermined (Exp 10)
        M=8, K=10, SNR_DB=15, T=1024, N_SCANS=30, proc_noise_deg=0.5,
        base_deg=list(np.linspace(-72, 72, 10)),
        rates_deg=[1.2, -0.8, 0.6, -1.0, 1.5, -1.3, 0.9, -0.7, 1.1, -1.4],
        pipelines=["COP-RFS", "COP-RFS-CL", "COP-RFS-GCL", "MUSIC-PHD", "MUSIC-PHD-CL"],
    ),
    "k10sw": dict(         # underdetermined: open vs closed-loop (MC) front-end, switch focus
        M=8, K=10, SNR_DB=15, T=256, N_SCANS=30, proc_noise_deg=0.5,
        base_deg=list(np.linspace(-72, 72, 10)),
        rates_deg=[1.2, -0.8, 0.6, -1.0, 1.5, -1.3, 0.9, -0.7, 1.1, -1.4],
        pipelines=["COP-RFS", "COP-RFS-MC"],
    ),
    "k6stat": dict(        # in-regime for closed loop: low-SNR, (near-)stationary
        M=8, K=6, SNR_DB=5, T=256, N_SCANS=20, proc_noise_deg=0.2,
        base_deg=[-50, -30, -10, 10, 30, 50],
        rates_deg=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        pipelines=["COP-RFS", "COP-RFS-CL", "COP-RFS-GCL", "MUSIC-PHD", "MUSIC-PHD-CL"],
    ),
    "k4": dict(            # determined crossing ablation (Table I)
        M=8, K=4, SNR_DB=15, T=512, N_SCANS=30, proc_noise_deg=0.5,
        base_deg=[-40, 40, -20, 20],
        rates_deg=[2.5, -2.5, 1.5, -1.5],
        pipelines=["COP+Physics", "COP+Standard", "MUSIC+Physics", "MUSIC+Standard"],
    ),
    "kcross": dict(        # closely-spaced MOVING: resolution regime (prior helps resolve)
        M=8, K=2, SNR_DB=5, T=256, N_SCANS=20, proc_noise_deg=0.5,
        base_deg=[-1.5, 1.5],          # 3 deg apart (near COP resolution)
        rates_deg=[1.0, 1.0],          # both drift +1 deg/scan (moving, stay close)
        pipelines=["COP-RFS", "COP-RFS-MC", "MUSIC-PHD"],
    ),
    "ablation": dict(      # tracking-filter ablation: GM-PHD vs TO-PHD vs LMB vs GLMB
        M=8, K=4, SNR_DB=15, T=512, N_SCANS=30, proc_noise_deg=0.5,
        base_deg=[-40, 40, -20, 20],
        rates_deg=[2.5, -2.5, 1.5, -1.5],
        pipelines=["COP+Standard", "COP+Physics", "COP+LMB", "COP+GLMB"],
    ),
    "compose": dict(       # do front-end loop and SOTA back-end gains compose? (crossing)
        M=8, K=4, SNR_DB=15, T=512, N_SCANS=30, proc_noise_deg=0.5,
        base_deg=[-40, 40, -20, 20],
        rates_deg=[2.5, -2.5, 1.5, -1.5],
        pipelines=["COP+LMB", "COP-MC+LMB", "COP+GLMB", "COP-MC+GLMB"],
    ),
    "compose_stat": dict(  # clean composition: low-SNR STATIONARY, CL front-end x SOTA back-end
        M=8, K=6, SNR_DB=5, T=256, N_SCANS=20, proc_noise_deg=0.2,
        base_deg=[-50, -30, -10, 10, 30, 50],
        rates_deg=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        pipelines=["COP+LMB", "COP-CL+LMB", "COP+GLMB", "COP-CL+GLMB"],
    ),
    "gate_cross": dict(    # GATE safety on the crossing where naive MC HURT: GCL must match open
        M=8, K=4, SNR_DB=15, T=512, N_SCANS=30, proc_noise_deg=0.5,
        base_deg=[-40, 40, -20, 20],
        rates_deg=[2.5, -2.5, 1.5, -1.5],
        pipelines=["COP+GLMB", "COP-MC+GLMB", "COP-GCL+GLMB"],
    ),
    "gate_stat": dict(     # GATE on the stationary regime where CL HELPED: GCL must still help
        M=8, K=6, SNR_DB=5, T=256, N_SCANS=20, proc_noise_deg=0.2,
        base_deg=[-50, -30, -10, 10, 30, 50],
        rates_deg=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        pipelines=["COP+GLMB", "COP-CL+GLMB", "COP-GCL+GLMB"],
    ),
    "k4move": dict(        # determined FAST crossing: gate-safety demo (stable tracks)
        M=8, K=4, SNR_DB=15, T=512, N_SCANS=30, proc_noise_deg=0.5,
        base_deg=[-40, 40, -20, 20],
        rates_deg=[2.5, -2.5, 1.5, -1.5],
        pipelines=["COP-RFS", "COP-RFS-CL", "COP-RFS-GCL", "COP-RFS-MC", "MUSIC-PHD"],
    ),
}
WARMUP = 3
GOSPA_C_DEG = 10.0
DET_THR_DEG = 3.0


# ===================================================================== #
#  정확한 지표                                                          #
# ===================================================================== #
def _ang(a, b):
    d = abs(a - b)
    return min(d, 2 * np.pi - d)


def loc_rmse_deg(est_rad, true_rad, c_rad):
    """GOSPA-매칭쌍(거리<c) 한정 localization RMSE (deg). 매칭 없으면 nan."""
    est = np.asarray(est_rad).ravel()
    tru = np.asarray(true_rad).ravel()
    if est.size == 0 or tru.size == 0:
        return np.nan
    D = np.abs(est[:, None] - tru[None, :])
    D = np.minimum(D, 2 * np.pi - D)            # wrap
    r, c = linear_sum_assignment(D)
    pairs = [(i, j) for i, j in zip(r, c) if D[i, j] < c_rad]
    if not pairs:
        return np.nan
    sq = np.mean([D[i, j] ** 2 for i, j in pairs])
    return float(np.degrees(np.sqrt(sq)))


def gospa_decomp(est_rad, true_rad, c_rad, p=2, alpha=2):
    """GOSPA + 성분 분해. 반환: (total_deg, loc_deg, missed_deg, false_deg)."""
    from iron_dome_sim.eval.metrics import gospa
    est = np.asarray(est_rad).ravel()
    tru = np.asarray(true_rad).ravel()
    e = est.reshape(-1, 1) if est.size else np.empty((0, 1))
    t = tru.reshape(-1, 1) if tru.size else np.empty((0, 1))
    g, dec = gospa(e, t, c=c_rad, p=p, alpha=alpha)
    return (np.degrees(g),
            np.degrees(dec.get("localization", 0.0)),
            np.degrees(dec.get("missed", 0.0)),
            np.degrees(dec.get("false", 0.0)))


def detrate(est_rad, true_rad, thr_rad):
    from iron_dome_sim.eval.metrics import detection_rate
    pd, _ = detection_rate(np.asarray(est_rad).ravel(),
                           np.asarray(true_rad).ravel(), thr_rad)
    return pd


def assign_labels(label_hist, K, base_rad, rates_rad, N):
    """PHD 라벨 -> 소스 인덱스 매핑 (원본 로직과 동일)."""
    all_labels = set()
    for tlh in label_hist:
        all_labels.update(tlh.keys())
    l2s = {}
    for label in sorted(all_labels):
        states, sids = [], []
        for si, tlh in enumerate(label_hist):
            if label in tlh:
                states.append(tlh[label][0])
                sids.append(si)
        if not states:
            continue
        best_k, best_c = -1, float("inf")
        for k in range(K):
            cost = sum(
                abs(s[0] - np.clip(base_rad[k] + rates_rad[k] * si,
                                   -np.pi / 2 + 0.05, np.pi / 2 - 0.05))
                + 2.0 * (abs(s[2] - rates_rad[k]) if len(s) > 2 else 0)
                for s, si in zip(states, sids)
            ) / len(states)
            if cost < best_c:
                best_c, best_k = cost, k
        l2s[label] = best_k
    return l2s


def count_switches(label_hist, l2s, K):
    src_labels = {k: [] for k in range(K)}
    for si, tlh in enumerate(label_hist):
        for label, tup in tlh.items():
            src = l2s.get(label, -1)
            if 0 <= src < K:
                src_labels[src].append((si, label))
    sw = 0
    for k in range(K):
        seq = src_labels[k]
        for i in range(1, len(seq)):
            if seq[i][1] != seq[i - 1][1]:
                sw += 1
    return sw


# ===================================================================== #
#  단일 trial 실행 (병렬 워커)                                          #
# ===================================================================== #
def build_estimator(name, array, K):
    from iron_dome_sim.doa import SubspaceCOP, MUSIC
    if "-MC" in name:                                     # 운동보상 T-COP (예측 prior, 누적 없음); 결합 pipe "COP-MC+GLMB" 포함
        from gated_tcop import MotionCompTCOP
        return MotionCompTCOP(array, rho=2, num_sources=K,
                              prior_weight=0.5, gate_delta2=0.3)
    if "GCL" in name:                                     # 게이트 폐루프 COP (G-T-COP)
        from gated_tcop import GatedTemporalCOP
        return GatedTemporalCOP(array, rho=2, num_sources=K, alpha=0.85,
                                prior_weight=0.5, gate_delta2=0.3, vel_thresh_deg=0.4)
    if "MUSIC" in name and "CL" in name:                  # 폐루프 MUSIC (T-MUSIC)
        from closed_loop_music import TemporalMUSIC
        return TemporalMUSIC(array, num_sources=min(K, array.M - 1),
                             alpha=0.85, prior_weight=0.3)
    if "CL" in name:                                       # 폐루프 T-COP
        from iron_dome_sim.doa.temporal_cop import TemporalCOP
        return TemporalCOP(array, rho=2, num_sources=K, alpha=0.85, prior_weight=0.3)
    if name.startswith("COP"):
        return SubspaceCOP(array, rho=2, num_sources=K, spectrum_type="combined")
    return MUSIC(array, num_sources=min(K, array.M - 1))   # MUSIC capped at M-1


def build_phd(estimator, use_physics):
    from iron_dome_sim.tracking import COPPHD, ConstantVelocity
    model = ConstantVelocity(dt=1.0, process_noise_std=np.radians(0.5))
    return COPPHD(model, estimator, survival_prob=0.95, detection_prob=0.95,
                  birth_weight=0.5, clutter_rate=0.3, prune_threshold=1e-3,
                  merge_threshold=2.0, birth_pos_std_deg=3.0,
                  birth_vel_std_deg=5.0, association_gate_deg=10.0,
                  use_physics=use_physics)


def run_trial(args):
    """args=(scenario_name, pipeline_name, trial_idx) -> per-trial scalar dict."""
    scen_name, pipe, trial = args
    sc = SCENARIOS[scen_name]
    from iron_dome_sim.signal_model.array import UniformLinearArray
    from iron_dome_sim.signal_model.signal_generator import generate_snapshots

    M, K, N = sc["M"], sc["K"], sc["N_SCANS"]
    T = int(os.environ.get("EVAL_T") or sc["T"])        # 워커는 env 로 override 수신
    snr = float(os.environ.get("EVAL_SNR") or sc["SNR_DB"])
    base = np.radians(sc["base_deg"])
    rates = np.radians(sc["rates_deg"])
    use_physics = not pipe.endswith("Standard")     # 'Standard' only in k4

    array = UniformLinearArray(M=M, d=0.5)
    est = build_estimator(pipe, array, K)
    if "GLMB" in pipe:                                 # delta-GLMB (full labeled RFS)
        from glmb_tracker import GLMBTracker            # NB: check before LMB ("LMB" in "GLMB")
        from iron_dome_sim.tracking import ConstantVelocity
        model = ConstantVelocity(dt=1.0, process_noise_std=np.radians(0.5))
        phd = GLMBTracker(model, est)
    elif "LMB" in pipe:                                 # 라벨드 멀티-베르누이 트래커
        from lmb_tracker import LMBTracker
        from iron_dome_sim.tracking import ConstantVelocity
        model = ConstantVelocity(dt=1.0, process_noise_std=np.radians(0.5))
        phd = LMBTracker(model, est)
    else:
        phd = build_phd(est, use_physics=use_physics)
    scan = np.linspace(-np.pi / 2, np.pi / 2, 1801)
    c_rad = np.radians(GOSPA_C_DEG)

    label_hist = []
    pd_s, loc_s, g_s, gl_s, gm_s, gf_s = [], [], [], [], [], []
    for si in range(N):
        true = np.clip(base + rates * si, -np.pi / 2 + 0.05, np.pi / 2 - 0.05)
        # 독립 시드: trial 마다 다른 잡음 실현
        np.random.seed(100000 * (trial + 1) + 42 + si)
        X, _, _ = generate_snapshots(array, true, snr, T, "non_stationary")
        phd.process_scan(X, scan)
        est_doa = np.asarray(phd.get_doa_estimates()).ravel()
        label_hist.append(phd.get_track_states())

        pd_s.append(detrate(est_doa, true, np.radians(DET_THR_DEG)))
        loc_s.append(loc_rmse_deg(est_doa, true, c_rad))
        gt, gl, gm, gf = gospa_decomp(est_doa, true, c_rad)
        g_s.append(gt); gl_s.append(gl); gm_s.append(gm); gf_s.append(gf)

    l2s = assign_labels(label_hist, K, base, rates, N)
    sw = count_switches(label_hist, l2s, K)

    def m(x):
        x = np.asarray(x[WARMUP:], float)
        x = x[np.isfinite(x)]
        return float(np.mean(x)) if x.size else np.nan

    return dict(pipeline=pipe, pd=m(pd_s) * 100, loc_rmse=m(loc_s),
                gospa=m(g_s), g_loc=m(gl_s), g_missed=m(gm_s),
                g_false=m(gf_s), switches=float(sw))


# ===================================================================== #
#  집계 + 출력                                                          #
# ===================================================================== #
def ci95(vals):
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan
    return float(np.mean(v)), float(1.96 * np.std(v, ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=list(SCENARIOS), default="k10")
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--T", type=int, default=0, help="스냅샷 수 override (0=시나리오 기본)")
    ap.add_argument("--snr", type=float, default=None, help="SNR(dB) override")
    a = ap.parse_args()

    sc = SCENARIOS[a.scenario]
    if a.T > 0:
        sc["T"] = a.T
        os.environ["EVAL_T"] = str(a.T)            # 워커 전달용 (spawn 시 상속)
    if a.snr is not None:
        sc["SNR_DB"] = a.snr
        os.environ["EVAL_SNR"] = str(a.snr)

    print("=" * 72)
    print(f"MC 평가 | scenario={a.scenario}  trials={a.trials}  "
          f"M={sc['M']} K={sc['K']} SNR={sc['SNR_DB']}dB T={sc['T']} "
          f"scans={sc['N_SCANS']}")
    print("=" * 72)

    tasks = [(a.scenario, pipe, t)
             for pipe in sc["pipelines"] for t in range(a.trials)]

    results = {p: [] for p in sc["pipelines"]}
    if a.jobs > 1:
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            for r in ex.map(run_trial, tasks):
                results[r["pipeline"]].append(r)
    else:
        for tk in tasks:
            results[tk[1]].append(run_trial(tk))

    hdr = f"{'Pipeline':<16} {'Pd(%)':>13} {'locRMSE(°)':>14} {'GOSPA(°)':>13} {'  [loc/miss/fa]':>20} {'Switch':>11}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for p in sc["pipelines"]:
        rs = results[p]
        pd_m, pd_e = ci95([r["pd"] for r in rs])
        lr_m, lr_e = ci95([r["loc_rmse"] for r in rs])
        g_m, g_e = ci95([r["gospa"] for r in rs])
        gl, _ = ci95([r["g_loc"] for r in rs])
        gm, _ = ci95([r["g_missed"] for r in rs])
        gf, _ = ci95([r["g_false"] for r in rs])
        sw_m, sw_e = ci95([r["switches"] for r in rs])
        print(f"{p:<16} {pd_m:6.1f}±{pd_e:4.1f} {lr_m:7.2f}±{lr_e:4.2f} "
              f"{g_m:7.2f}±{g_e:4.2f}  {gl:4.1f}/{gm:4.1f}/{gf:4.1f}  "
              f"{sw_m:6.1f}±{sw_e:4.1f}")
    print("\n(±는 95% 신뢰구간. locRMSE=GOSPA매칭쌍 한정 위치오차; "
          "GOSPA 성분=loc/missed/false)")


if __name__ == "__main__":
    main()
