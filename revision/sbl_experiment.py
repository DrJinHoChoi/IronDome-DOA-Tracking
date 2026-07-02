# -*- coding: utf-8 -*-
"""
SBL baseline on the underdetermined end-to-end scenario (k10, K=10 > M-1=7).

Runs a genuine multisnapshot Sparse Bayesian Learning (SBL) front-end through the
SAME TO-PHD back-end as COP-RFS / MUSIC-PHD, under the identical k10 scenario and
per-trial seeds (paired comparison). Reuses eval_mc.run_trial verbatim, so the
COP-RFS / MUSIC-PHD columns reproduce Table (underdetermined) as a sanity check.

Purpose: keep SBL results in hand (Reviewer R2.5) without touching the manuscript.

  python revision/sbl_experiment.py                 # NTRIALS default (quick)
  NTRIALS=500 JOBS=8 python revision/sbl_experiment.py    # match headline table
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

PIPES = ["SBL-PHD", "COP-RFS", "MUSIC-PHD"]
N_TRIALS = int(os.environ.get("NTRIALS", "60"))
SCEN = "k10"


def ci(v):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    if v.size == 0:
        return float("nan"), float("nan")
    m = float(np.mean(v))
    h = float(1.96 * np.std(v, ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0
    return m, h


def main():
    sc = E.SCENARIOS[SCEN]
    print(f"scenario={SCEN}  K={sc['K']}>{sc['M']-1}  SNR={sc['SNR_DB']}dB  "
          f"T={sc['T']}  scans={sc['N_SCANS']}  trials={N_TRIALS}")
    tasks = [(SCEN, p, t) for p in PIPES for t in range(N_TRIALS)]
    acc = {p: {"pd": [], "loc": [], "gospa": [], "sw": []} for p in PIPES}
    jobs = int(os.environ.get("JOBS", "8"))
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for r in ex.map(E.run_trial, tasks):
            p = r["pipeline"]
            acc[p]["pd"].append(r["pd"]); acc[p]["loc"].append(r["loc_rmse"])
            acc[p]["gospa"].append(r["gospa"]); acc[p]["sw"].append(r["switches"])

    print("\n=== Underdetermined end-to-end (K=10 > M-1=7): SBL vs references ===")
    print(f"{'Pipeline':10} | {'Pd (%)':>13} | {'locRMSE(deg)':>14} | {'GOSPA(deg)':>13} | {'Switches':>12}")
    print("-" * 74)
    for p in PIPES:
        pd, loc, g, sw = (ci(acc[p][k]) for k in ("pd", "loc", "gospa", "sw"))
        print(f"{p:10} | {pd[0]:6.1f} +-{pd[1]:4.1f} | {loc[0]:6.2f} +-{loc[1]:4.2f} | "
              f"{g[0]:6.2f} +-{g[1]:4.2f} | {sw[0]:6.1f} +-{sw[1]:4.1f}")
    print("\n(COP-RFS / MUSIC-PHD should match the manuscript's underdetermined table.)")


if __name__ == "__main__":
    main()
