# -*- coding: utf-8 -*-
"""
Source-count (K) sweep: robustness of the end-to-end pipeline across the number
of sources, spanning the determined (K<=M-1) and underdetermined (K>M-1) regimes.

M=8 (M-1=7). K in {2,4,6,8,10,12}. Pipelines: MUSIC-PHD (capped at M-1),
COP-RFS (open-loop), COP-RFS-GCL (deployed gated closed loop).
Metrics: detection rate Pd, GOSPA, identity switches.  Output: fig_k_sweep.png.

복수 K 에서 폐루프/COP 이득이 유지되는지(특정 K cherry-pick 아님)를 보인다.
"""
from __future__ import annotations
import os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
import eval_mc as E   # reuse build_estimator/build_phd + metrics

M = 8
T = 512
N_SCANS = 15
N_TRIALS = int(os.environ.get("NTRIALS", "20"))
SNR = 12
KVALS = [2, 4, 6, 8, 10, 12]
PIPES = ["MUSIC-PHD", "COP-RFS", "COP-RFS-GCL"]


def _targets(K):
    base = np.radians(np.linspace(-60.0, 60.0, K))
    rates = np.zeros(K)          # (near-)stationary: isolate source-count effect; GCL engages cleanly
    return base, rates


def run_trial(args):
    import numpy as np
    from iron_dome_sim.signal_model.array import UniformLinearArray
    from iron_dome_sim.signal_model.signal_generator import generate_snapshots
    K, pipe, trial = args
    base, rates = _targets(K)
    array = UniformLinearArray(M=M, d=0.5)
    est = E.build_estimator(pipe, array, K)
    phd = E.build_phd(est, use_physics=True)
    scan = np.linspace(-np.pi / 2, np.pi / 2, 1801)
    c = np.radians(E.GOSPA_C_DEG)
    pd_s, g_s, label_hist = [], [], []
    for si in range(N_SCANS):
        true = np.clip(base + rates * si, -np.pi / 2 + 0.05, np.pi / 2 - 0.05)
        np.random.seed(30000 * (trial + 1) + K * 131 + si)
        X, _, _ = generate_snapshots(array, true, SNR, T, "non_stationary")
        phd.process_scan(X, scan)
        ed = np.asarray(phd.get_doa_estimates()).ravel()
        label_hist.append(phd.get_track_states())
        pd_s.append(E.detrate(ed, true, np.radians(E.DET_THR_DEG)))
        gt, gl, gm, gf = E.gospa_decomp(ed, true, c)
        g_s.append(gt)
    l2s = E.assign_labels(label_hist, K, base, rates, N_SCANS)
    sw = E.count_switches(label_hist, l2s, K)

    def m(x):
        x = np.asarray(x[E.WARMUP:], float)
        x = x[np.isfinite(x)]
        return float(np.mean(x)) if x.size else np.nan
    return K, pipe, m(pd_s) * 100.0, m(g_s), float(sw)


def ci(v):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    if v.size == 0:
        return np.nan, np.nan
    return float(np.mean(v)), (float(1.96 * np.std(v, ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0)


def main():
    tasks = [(K, p, t) for K in KVALS for p in PIPES for t in range(N_TRIALS)]
    accpd = {(K, p): [] for K in KVALS for p in PIPES}
    accg = {(K, p): [] for K in KVALS for p in PIPES}
    accsw = {(K, p): [] for K in KVALS for p in PIPES}
    jobs = int(os.environ.get("JOBS", "8"))
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for K, p, pd, g, sw in ex.map(run_trial, tasks):
            accpd[(K, p)].append(pd)
            accg[(K, p)].append(g)
            accsw[(K, p)].append(sw)

    print("=== K-sweep (M=8, M-1=7): Pd(%) / GOSPA(deg) / switches ===")
    for K in KVALS:
        print(f"K={K:2d}: " + " | ".join(
            f"{p} Pd={ci(accpd[(K,p)])[0]:4.1f} G={ci(accg[(K,p)])[0]:5.2f} "
            f"sw={ci(accsw[(K,p)])[0]:4.1f}" for p in PIPES))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9})
    sty = {"MUSIC-PHD": ("#d62728", "-s", "MUSIC--PHD"),
           "COP-RFS": ("#1f77b4", "-o", "COP-RFS (open-loop)"),
           "COP-RFS-GCL": ("#2ca02c", "-.^", "COP-RFS gated (proposed)")}
    # single column-native panel: detection P_d vs K (GOSPA-vs-K story is redundant)
    fig, ax = plt.subplots(figsize=(3.45, 2.75))
    for p in PIPES:
        col, fmt, lab = sty[p]
        pdv = [ci(accpd[(K, p)])[0] for K in KVALS]
        pde = [ci(accpd[(K, p)])[1] for K in KVALS]
        ax.errorbar(KVALS, pdv, yerr=pde, fmt=fmt, color=col, capsize=2.5,
                    lw=1.6, ms=5, label=lab)
    ax.axvline(M - 1, ls=":", color="gray")
    ax.text(M - 1 + 0.12, 45, "$M{-}1$", color="gray", fontsize=8.5)
    ax.set_xlabel("number of sources $K$")
    ax.set_ylabel("detection rate $P_d$ (%)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.2, loc="lower left")
    fig.tight_layout(pad=0.3)
    out = os.path.join(HERE, "fig_k_sweep.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    main()
