# SBL baseline — results in hand (Reviewer R2.5)

**Status:** kept for rebuttal only. The submitted manuscript is unchanged. If a
reviewer insists on a quantitative SBL comparison, drop the row below into the
underdetermined table and use the rebuttal paragraph.

## What was implemented
A *genuine* multisnapshot Sparse Bayesian Learning (SBL) DOA front-end
(`iron_dome_sim/doa/sbl.py`) — Gerstoft-style multiplicative evidence
maximization over per-grid-point powers, **not** an l1 sparse-recovery surrogate.
It is run through the **same TO-PHD back-end** as COP-RFS / MUSIC-PHD, on the
**same** underdetermined scenario and **paired per-trial seeds**.

Reproduce:
```
NTRIALS=24 JOBS=6 python revision/sbl_experiment.py     # (500 for the headline setting)
```

## Result (k10: K=10 > M-1=7, M=8 ULA, SNR=15 dB, T=1024, 30 scans, 24 trials, 95% CI)

| Pipeline (front-end) | Pd (%) | locRMSE (deg) | GOSPA (deg) | Switches |
|---|---|---|---|---|
| **COP-RFS (4th-order)** | **60.6 ± 0.6** | 3.00 ± 0.10 | 16.67 ± 0.20 | 195.5 ± 11.0 |
| MUSIC-PHD (2nd-order)   | 51.9 ± 0.4 | 2.60 ± 0.05 | 15.40 ± 0.06 | 63.0 ± 6.4 |
| **SBL-PHD (2nd-order sparse)** | 38.1 ± 0.4 | **1.72 ± 0.03** | 17.32 ± 0.05 | 124.7 ± 6.7 |

(COP-RFS and MUSIC-PHD reproduce the manuscript's underdetermined table — same harness/conditions.)

## Interpretation (this *strengthens* the paper)
- **The higher-order COP front-end significantly out-detects both second-order
  baselines** (60.6% vs 51.9% MUSIC and 38.1% SBL; non-overlapping CIs).
- **SBL is bounded by the same M-1 aperture limit as MUSIC.** On a *filled* ULA
  the second-order difference co-array does not extend the degrees of freedom
  beyond M-1, so no second-order method — subspace (MUSIC) or sparse (SBL) — can
  resolve K > M-1. Only the **fourth-order** cumulant COP (virtual aperture
  M_v = rho(M-1)+1 = 15) breaks the cap. The underdetermined gain is therefore
  intrinsically *higher-order*, not obtainable from sparsity on the same array.
- **SBL attains the sharpest localization (1.72 deg)** but the fewest detections
  — the classic sparse-method trade-off, and evidence the implementation is sound
  (it is a well-behaved SBL, not a broken one).

## Manuscript-ready table row (if adding to the underdetermined table)
```latex
SBL--PHD & $38.1\pm0.4$ & $\mathbf{1.72\pm0.03}$ & $17.32\pm0.05$ & $125\pm7$ \\
```

## Rebuttal paragraph (ready to paste)
> As requested, we implemented a genuine multisnapshot Sparse Bayesian Learning
> (SBL) DOA front-end (Gerstoft-style evidence maximization) and evaluated it
> through the identical TO-PHD back-end on the underdetermined scenario
> (K=10 > M-1=7). SBL attains the sharpest localization (1.72 deg RMSE) but the
> fewest detections (Pd = 38.1% +- 0.4), below both MUSIC (51.9%) and the
> higher-order COP (60.6%). This is expected and, in fact, reinforces our thesis:
> on a filled ULA the second-order difference co-array does not extend the
> degrees of freedom beyond M-1, so no second-order method — subspace or sparse —
> resolves K > M-1; the underdetermined capability is intrinsically fourth-order.
> Accordingly we treat SBL as a second-order sparse baseline confirming the
> aperture cap, not as a competitor in the underdetermined regime.
