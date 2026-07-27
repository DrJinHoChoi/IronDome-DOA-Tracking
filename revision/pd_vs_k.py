# -*- coding: utf-8 -*-
"""
Pd-vs-K 프론트엔드 검출률 스윕 (구조적 한계 시각화).

M=8 고정, K=3..14 소스를 well-separated 배치, 각 K에서 COP-4th 와 MUSIC 의
스캔당 검출률(3deg 이내)을 N회 평균 + 95% CI.

기대: MUSIC 은 K>M-1=7 에서 (M-1)/K 천장(=7/K)으로 정체/하락,
COP 는 가상배열(M_v=15)로 K=14 까지 높게 유지 -> 구조적 우위가 한눈에.

출력: revision/fig_pd_vs_k.png  (+ figures/ 로 복사)
"""
from __future__ import annotations
import os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

M = 8
RHO = 2
SNR_DB = 15.0
T = 512
KS = list(range(3, 15))          # 3..14
N_TRIALS = 60
FOV = (-60.0, 60.0)
SCAN = None                      # set in worker
DET_DEG = 3.0


def run_trial(args):
    K, trial = args
    import numpy as np
    from iron_dome_sim.signal_model.array import UniformLinearArray
    from iron_dome_sim.signal_model.signal_generator import generate_snapshots
    from iron_dome_sim.doa import SubspaceCOP, MUSIC
    from iron_dome_sim.eval.metrics import detection_rate

    array = UniformLinearArray(M=M, d=0.5)
    thetas = np.deg2rad(np.linspace(FOV[0], FOV[1], K))
    scan = np.linspace(-np.pi / 2, np.pi / 2, 1201)
    np.random.seed(7000 * (trial + 1) + K)
    X, _, _ = generate_snapshots(array, thetas, SNR_DB, T, "non_stationary")

    cop = SubspaceCOP(array, rho=RHO, num_sources=K, spectrum_type="combined")
    music = MUSIC(array, num_sources=min(K, M - 1))
    cop_doa, _ = cop.estimate(X, scan)
    mus_doa, _ = music.estimate(X, scan)
    pd_cop, _ = detection_rate(np.asarray(cop_doa).ravel(), thetas, np.radians(DET_DEG))
    pd_mus, _ = detection_rate(np.asarray(mus_doa).ravel(), thetas, np.radians(DET_DEG))
    return K, pd_cop * 100, pd_mus * 100


def ci95(v):
    v = np.asarray(v, float)
    return float(np.mean(v)), float(1.96 * np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0


def main():
    tasks = [(K, t) for K in KS for t in range(N_TRIALS)]
    res = {K: {"cop": [], "mus": []} for K in KS}
    jobs = int(os.environ.get("JOBS", "8"))
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for K, pc, pm in ex.map(run_trial, tasks):
            res[K]["cop"].append(pc)
            res[K]["mus"].append(pm)

    print(f"{'K':>3} {'COP Pd%':>14} {'MUSIC Pd%':>14} {'(M-1)/K cap%':>13}")
    cop_m, cop_e, mus_m, mus_e = [], [], [], []
    for K in KS:
        cm, ce = ci95(res[K]["cop"]); mm, me = ci95(res[K]["mus"])
        cop_m.append(cm); cop_e.append(ce); mus_m.append(mm); mus_e.append(me)
        print(f"{K:>3} {cm:7.1f}±{ce:4.1f} {mm:7.1f}±{me:4.1f} {100*min(M-1,K)/K:12.1f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HERE = os.path.dirname(os.path.abspath(__file__))
    KSa = np.array(KS)
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.errorbar(KSa, cop_m, yerr=cop_e, fmt="-o", color="#1f77b4", capsize=3,
                lw=2, label="COP-4th (proposed)")
    ax.errorbar(KSa, mus_m, yerr=mus_e, fmt="-s", color="#d62728", capsize=3,
                lw=2, label="MUSIC")
    ax.plot(KSa, [100 * min(M - 1, k) / k for k in KS], "k--", lw=1.3,
            label=r"MUSIC capacity $(M{-}1)/K$")
    ax.axvline(M - 1, color="gray", ls=":", lw=1.2)
    ax.text(M - 1 + 0.1, 8, r"$M{-}1{=}7$", color="gray", fontsize=9)
    ax.axvline(RHO * (M - 1), color="green", ls=":", lw=1.2)
    ax.text(RHO * (M - 1) - 1.7, 8, r"$\rho(M{-}1){=}14$", color="green", fontsize=9)
    ax.set_xlabel("number of sources  K")
    ax.set_ylabel("front-end detection rate (%)")
    ax.set_title(f"Detection vs. K  (M={M}, SNR={int(SNR_DB)} dB, T={T})")
    ax.set_ylim([0, 105]); ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout()
    out = os.path.join(HERE, "fig_pd_vs_k.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    main()
