# -*- coding: utf-8 -*-
"""
Signal-type (HOS) robustness: does the higher-order-cumulant front-end keep its
underdetermined capability across diverse source signal types?

K=10 (> M-1=7, underdetermined: COP's domain), M=8.  Sweep the source signal type
over realistic non-Gaussian radar/comm waveforms {non_stationary, missile, fm,
chirp, psk} plus the pure complex-Gaussian case (stationary).  Pipelines: COP-RFS
(4th-order) and MUSIC-PHD (2nd-order, capped at M-1).  Metric: detection Pd, GOSPA.

기대(정직한 HOS 특성화):
  - 비Gaussian(레이더/통신) 소스: COP 가 K>M-1 검출 유지 (4차 cumulant 비소멸).
  - 순수 Gaussian("stationary"): 4차 cumulant ~ 0 -> COP 원리상 실패(한계).
  - MUSIC 는 신호종류 무관하나 M-1 에 포화.
출력: 표(stdout) + fig_sigtype.png
"""
from __future__ import annotations
import os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
import eval_mc as E

M = 8
K = 10
T = 1024
N_SCANS = 12
N_TRIALS = int(os.environ.get("NTRIALS", "20"))
SNR = 15
SIGTYPES = ["non_stationary", "missile", "fm", "chirp", "psk", "stationary"]
LABELS = {"non_stationary": "AM radar", "missile": "missile\n(Doppler+RCS)",
          "fm": "FM", "chirp": "LFM chirp", "psk": "PSK comm",
          "stationary": "Gaussian\n(HOS limit)"}
PIPES = ["COP-RFS", "MUSIC-PHD"]
BASE = np.radians(np.linspace(-72.0, 72.0, K))
RATES = np.radians(np.array([0.4 * (1 if i % 2 else -1) for i in range(K)]))


def run_trial(args):
    import numpy as np
    from iron_dome_sim.signal_model.array import UniformLinearArray
    from iron_dome_sim.signal_model.signal_generator import generate_snapshots
    sig, pipe, trial = args
    array = UniformLinearArray(M=M, d=0.5)
    est = E.build_estimator(pipe, array, K)
    phd = E.build_phd(est, use_physics=True)
    scan = np.linspace(-np.pi / 2, np.pi / 2, 1801)
    c = np.radians(E.GOSPA_C_DEG)
    pd_s, g_s = [], []
    for si in range(N_SCANS):
        true = np.clip(BASE + RATES * si, -np.pi / 2 + 0.05, np.pi / 2 - 0.05)
        np.random.seed(40000 * (trial + 1) + hash(sig) % 9973 + si)
        X, _, _ = generate_snapshots(array, true, SNR, T, sig)
        phd.process_scan(X, scan)
        ed = np.asarray(phd.get_doa_estimates()).ravel()
        pd_s.append(E.detrate(ed, true, np.radians(E.DET_THR_DEG)))
        gt, gl, gm, gf = E.gospa_decomp(ed, true, c)
        g_s.append(gt)

    def m(x):
        x = np.asarray(x[E.WARMUP:], float)
        x = x[np.isfinite(x)]
        return float(np.mean(x)) if x.size else np.nan
    return sig, pipe, m(pd_s) * 100.0, m(g_s)


def ci(v):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    if v.size == 0:
        return np.nan, np.nan
    return float(np.mean(v)), (float(1.96 * np.std(v, ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0)


def main():
    tasks = [(s, p, t) for s in SIGTYPES for p in PIPES for t in range(N_TRIALS)]
    accpd = {(s, p): [] for s in SIGTYPES for p in PIPES}
    accg = {(s, p): [] for s in SIGTYPES for p in PIPES}
    jobs = int(os.environ.get("JOBS", "8"))
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for s, p, pd, g in ex.map(run_trial, tasks):
            accpd[(s, p)].append(pd)
            accg[(s, p)].append(g)

    print(f"=== signal-type HOS sweep (K={K}>M-1={M-1}, SNR={SNR}dB): Pd(%) / GOSPA(deg) ===")
    for s in SIGTYPES:
        print(f"{s:16s}: " + " | ".join(
            f"{p} Pd={ci(accpd[(s,p)])[0]:4.1f}+-{ci(accpd[(s,p)])[1]:.1f} "
            f"G={ci(accg[(s,p)])[0]:5.2f}" for p in PIPES))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.arange(len(SIGTYPES))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    for i, p in enumerate(PIPES):
        vals = [ci(accpd[(s, p)])[0] for s in SIGTYPES]
        errs = [ci(accpd[(s, p)])[1] for s in SIGTYPES]
        col = "#1f77b4" if p == "COP-RFS" else "#d62728"
        lab = "COP-RFS (4th-order)" if p == "COP-RFS" else "MUSIC--PHD (2nd-order)"
        ax.bar(x + (i - 0.5) * w, vals, w, yerr=errs, capsize=3, color=col, label=lab)
    ax.axhline(100.0 * (M - 1) / K, ls=":", color="gray")
    ax.text(len(SIGTYPES) - 1.0, 100.0 * (M - 1) / K + 1.5,
            "$(M{-}1)/K$ cap", color="gray", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[s] for s in SIGTYPES], fontsize=8)
    ax.set_ylabel("detection rate $P_d$ (%)")
    ax.set_title(f"Underdetermined $K={K}>M{{-}}1={M-1}$: HOS holds for non-Gaussian sources")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(HERE, "fig_sigtype.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    main()
