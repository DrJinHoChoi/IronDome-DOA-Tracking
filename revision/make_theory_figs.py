# -*- coding: utf-8 -*-
"""
Theory-validation figure, double-column (single 1x3 row, IEEE figure*).
Regenerates one combined figure whose per-panel generators were lost:

  fig_theory_validation.png -- three panels in a row:
    (a) Theorem 1 (closed-loop posterior CRB): worst-case angle variance vs
        source count K; the open-loop (data-only) bound diverges at the
        virtual-array limit K = M_v-1, the closed-loop PCRB stays finite.
    (b) Theorem 2, unbiased prior: fused/data MSE ratio = 1/(1+kappa).
    (c) Theorem 2, biased prior: ratio crosses 1 at the break-even
        ||b||^2 = D (sigma_e^2 + sigma_p^2); the gate keeps the deployed
        estimator at or below the data-only MSE.

Analytical/deterministic (a small seeded Monte-Carlo overlay), so the validated
structural facts are identical to the text.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DEG2 = (180.0 / np.pi) ** 2
plt.rcParams.update({"font.size": 8.5})


def _virtual_steer(M_v, thetas):
    n = np.arange(M_v)[:, None]
    ph = np.pi * n * np.sin(thetas)[None, :]
    A = np.exp(1j * ph)
    D = A * (1j * np.pi * n * np.cos(thetas)[None, :])
    return A, D


def _data_fim(thetas, M_v, sigma2, Teff):
    A, D = _virtual_steer(M_v, thetas)
    AhA = A.conj().T @ A
    P_perp = np.eye(M_v) - A @ np.linalg.pinv(AhA) @ A.conj().T
    R = A @ A.conj().T + sigma2 * np.eye(M_v)
    Rinv = np.linalg.inv(R)
    J = 2.0 * Teff * np.real((D.conj().T @ P_perp @ D) * (A.conj().T @ Rinv @ A).T)
    return 0.5 * (J + J.T)


def _panel_pcrb(ax):
    M, rho = 4, 2
    M_v = rho * (M - 1) + 1                            # = 7 ; virtual limit K = 6
    Kvals = [2, 3, 4, 5, 6, 7, 8]
    sigma2, Teff = 0.5, 80.0
    priors_deg = [0.5, 1.0, 2.0]
    BIG = 1e6
    open_v, closed = [], {s: [] for s in priors_deg}
    for K in Kvals:
        th = np.radians(np.linspace(-50.0, 50.0, K))
        J = _data_fim(th, M_v, sigma2, Teff)
        lmin = np.linalg.eigvalsh(J)[0]
        open_v.append(min(1.0 / lmin * DEG2, BIG) if lmin > 1e-9 else BIG)
        for s in priors_deg:
            Jp = J + np.eye(K) / np.radians(s) ** 2
            closed[s].append(1.0 / np.linalg.eigvalsh(Jp)[0] * DEG2)
    ax.semilogy(Kvals, open_v, "-s", color="k", lw=1.6, ms=4.5,
                label="open-loop (data only)")
    for s, c in zip(priors_deg, ["#1f77b4", "#ff7f0e", "#2ca02c"]):
        ax.semilogy(Kvals, closed[s], "--o", color=c, lw=1.4, ms=3.5,
                    label=fr"closed, $\sigma_\theta={s:.1f}^\circ$")
    ax.axvline(M_v - 1, ls=":", color="#d62728", lw=1.3)
    ax.text(M_v - 1 - 0.12, 3e4, fr"virtual limit $K{{=}}M_v{{-}}1$",
            color="#d62728", fontsize=6.6, rotation=90, va="top", ha="right")
    ax.set_xlabel(r"sources $K$ (fixed FOV)")
    ax.set_ylabel(r"worst-case angle var. [deg$^2$]")
    ax.set_ylim(3e-2, 3e6)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=6.0, loc="center left", labelspacing=0.25, handlelength=1.6)
    ax.set_title(r"(a) Thm 1: PCRB identifiability", fontsize=8.5)


def _panel_unbiased(ax, rng):
    kap = np.logspace(-2, 2, 200)
    ax.plot(kap, 1.0 / (1.0 + kap), "-", color="#ff7f0e", lw=1.7,
            label=r"$1/(1+\kappa)$")
    kap_mc = np.logspace(-2, 2, 20)
    se2 = 1.0
    ratio = []
    for k in kap_mc:
        sp2 = se2 / k
        xe = rng.normal(0, np.sqrt(se2), 6000)
        xp = rng.normal(0, np.sqrt(sp2), 6000)
        fused = (xe / se2 + xp / sp2) / (1.0 / se2 + 1.0 / sp2)
        ratio.append(np.mean(fused ** 2) / se2)
    ax.plot(kap_mc, ratio, "o", color="#1f77b4", ms=3.6, label="Monte-Carlo")
    ax.axvline(1.0, ls=":", color="#d62728", lw=1.2)
    ax.text(1.3, 0.62, "prior = data\n(MSE halved)", color="#d62728", fontsize=6.4,
            va="center")
    ax.set_xscale("log")
    ax.set_xlabel(r"prior quality $\kappa=\sigma_e^2/\sigma_p^2$")
    ax.set_ylabel("fused / data-only MSE")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=6.6, loc="upper right")
    ax.set_title("(b) Thm 2: unbiased prior", fontsize=8.5)


def _panel_biased(ax, rng):
    beta = np.linspace(0.0, 3.0, 200)
    fused_cf = 0.25 * (2.0 * beta) + 0.5              # = 0.5*beta + 0.5
    ax.plot(beta, fused_cf, "-", color="#ff7f0e", lw=1.7, label="fused")
    beta_mc = np.linspace(0.0, 3.0, 22)
    fused_mc = []
    for b2n in beta_mc:
        b = np.sqrt(2.0 * b2n)
        xe = rng.normal(0, 1.0, 6000)
        xp = rng.normal(b, 1.0, 6000)
        fused_mc.append(np.mean((0.5 * xe + 0.5 * xp) ** 2))
    ax.plot(beta_mc, fused_mc, "o", color="#1f77b4", ms=3.6, label="fused (MC)")
    ax.plot(beta, np.minimum(fused_cf, 1.0), "-.", color="#2ca02c", lw=1.7,
            label="gated")
    ax.axhline(1.0, ls=":", color="k", lw=1.0)
    ax.axvline(1.0, ls=":", color="#d62728", lw=1.2)
    ax.text(1.06, 1.72, "break-even", color="#d62728", fontsize=6.4, va="center")
    ax.set_xlabel(r"norm. sq. bias $\|b\|^2/[D(\sigma_e^2+\sigma_p^2)]$")
    ax.set_ylabel("MSE / data-only MSE")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6.6, loc="upper left")
    ax.set_title("(c) Thm 2: biased prior + gate", fontsize=8.5)


def main():
    rng = np.random.RandomState(12345)
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(7.16, 2.45))
    _panel_pcrb(a1)
    _panel_unbiased(a2, rng)
    _panel_biased(a3, rng)
    fig.tight_layout(pad=0.4, w_pad=1.2)
    out = os.path.join(HERE, "fig_theory_validation.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    main()
