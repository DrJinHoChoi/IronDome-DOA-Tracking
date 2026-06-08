# -*- coding: utf-8 -*-
"""
결정역(K=6 < M-1=7, MUSIC 가능) SNR 스윕: 폐루프가 MUSIC-가능 영역에서도 동작하는가.

K=6 (near-)stationary, M=8. SNR in [-5..20] dB. 2x2 (COP/MUSIC x open/closed).
metric: matched-pair localization RMSE (deg), N trials, 95% CI.

기대(Theorem 2): 저SNR(큰 sigma_e) -> 폐루프 이득 큼; 고SNR -> 이득 수렴(0으로).
MUSIC < COP (결정역, 2차 분산 우위)이지만 폐루프는 둘 다 개선.

출력: revision/fig_snr_determined.png (+ figures/ 복사)
"""
from __future__ import annotations
import os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
import eval_mc as E   # reuse build_estimator / build_phd / loc_rmse_deg

K = 6
M = 8
T = 256
N_SCANS = 15
N_TRIALS = 30
SNRS = [-5, 0, 5, 10, 15, 20]
PIPES = ["COP-RFS", "COP-RFS-CL", "COP-RFS-GCL", "MUSIC-PHD", "MUSIC-PHD-CL"]
BASE = np.radians([-50, -30, -10, 10, 30, 50])
RATES = np.zeros(K)


def run_trial(args):
    import numpy as np
    from iron_dome_sim.signal_model.array import UniformLinearArray
    from iron_dome_sim.signal_model.signal_generator import generate_snapshots
    snr, pipe, trial = args
    use_physics = not pipe.endswith("Standard")
    array = UniformLinearArray(M=M, d=0.5)
    est = E.build_estimator(pipe, array, K)
    phd = E.build_phd(est, use_physics)
    scan = np.linspace(-np.pi / 2, np.pi / 2, 1201)
    c = np.radians(E.GOSPA_C_DEG)
    locs = []
    for si in range(N_SCANS):
        true = np.clip(BASE + RATES * si, -np.pi / 2 + 0.05, np.pi / 2 - 0.05)
        np.random.seed(20000 * (trial + 1) + int(round(snr)) * 31 + si)
        X, _, _ = generate_snapshots(array, true, snr, T, "non_stationary")
        phd.process_scan(X, scan)
        ed = np.asarray(phd.get_doa_estimates()).ravel()
        locs.append(E.loc_rmse_deg(ed, true, c))
    locs = [x for x in locs[3:] if np.isfinite(x)]
    return snr, pipe, (float(np.mean(locs)) if locs else np.nan)


def ci95(v):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    if v.size == 0:
        return np.nan, np.nan
    return float(np.mean(v)), float(1.96 * np.std(v, ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0


def main():
    tasks = [(s, p, t) for s in SNRS for p in PIPES for t in range(N_TRIALS)]
    acc = {(s, p): [] for s in SNRS for p in PIPES}
    jobs = int(os.environ.get("JOBS", "8"))
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for s, p, v in ex.map(run_trial, tasks):
            acc[(s, p)].append(v)

    print(f"{'SNR':>4} " + " ".join(f"{p:>14}" for p in PIPES))
    curves = {p: [] for p in PIPES}
    errs = {p: [] for p in PIPES}
    for s in SNRS:
        row = []
        for p in PIPES:
            m, e = ci95(acc[(s, p)])
            curves[p].append(m); errs[p].append(e)
            row.append(f"{m:6.2f}±{e:4.2f}")
        print(f"{s:>4} " + " ".join(f"{r:>14}" for r in row))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sty = {"COP-RFS": ("#1f77b4", "-o", "COP open-loop"),
           "COP-RFS-CL": ("#1f77b4", "--o", "COP closed-loop (full)"),
           "COP-RFS-GCL": ("#2ca02c", "-.^", "COP gated (deployed)"),
           "MUSIC-PHD": ("#d62728", "-s", "MUSIC open-loop"),
           "MUSIC-PHD-CL": ("#d62728", "--s", "MUSIC closed-loop")}
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for p in PIPES:
        col, fmt, lab = sty[p]
        filled = "--" not in fmt
        ax.errorbar(SNRS, curves[p], yerr=errs[p], fmt=fmt, color=col, capsize=3,
                    lw=2, markerfacecolor=(col if filled else "white"), label=lab)
    ax.set_yscale("log")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("localization RMSE (deg, log)")
    ax.set_title(f"Determined regime $K={K}<M{{-}}1=7$: closed loop helps both")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout()
    out = os.path.join(HERE, "fig_snr_determined.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    main()
