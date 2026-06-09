# -*- coding: utf-8 -*-
"""
Theory-validation figures, column-native (single IEEE column, readable fonts).
Regenerates two figures whose original generators were lost:

  fig_pcrb_validation.png  -- Theorem 1 (closed-loop posterior CRB):
      worst-case angle variance vs source count K. The open-loop (data-only)
      bound diverges at the virtual-array limit K = M_v-1, while the closed-loop
      PCRB stays finite, bounded by the prior information 1/sigma_theta^2.

  fig_tcop_mse_validation.png -- Theorem 2 (minimum-variance T-COP refinement):
      (a) unbiased prior: fused/data MSE ratio = 1/(1+kappa);
      (b) biased prior: ratio crosses 1 exactly at the break-even
          ||b||^2 = D (sigma_e^2 + sigma_p^2); the gate keeps the deployed
          estimator at or below the data-only MSE.

Both are analytical/deterministic (a small seeded Monte-Carlo overlay on the
T-COP panels), so the validated structural facts are identical to the text.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DEG2 = (180.0 / np.pi) ** 2
plt.rcParams.update({"font.size": 9})


# ---------------------------------------------------------------------------
# Theorem 1: closed-loop posterior CRB dominance / identifiability
# ---------------------------------------------------------------------------
def _virtual_steer(M_v, thetas):
    """Half-wavelength virtual ULA manifold and its angle derivative."""
    n = np.arange(M_v)[:, None]                     # M_v x 1
    ph = np.pi * n * np.sin(thetas)[None, :]        # M_v x K
    A = np.exp(1j * ph)
    D = A * (1j * np.pi * n * np.cos(thetas)[None, :])
    return A, D


def _data_fim(thetas, M_v, sigma2, Teff):
    """Deterministic (Stoica--Nehorai) DOA Fisher information, R_s = I."""
    A, D = _virtual_steer(M_v, thetas)
    K = A.shape[1]
    AhA = A.conj().T @ A
    # noise-subspace projector via pseudo-inverse (robust as K -> M_v and beyond)
    P_perp = np.eye(M_v) - A @ np.linalg.pinv(AhA) @ A.conj().T
    R = A @ A.conj().T + sigma2 * np.eye(M_v)
    Rinv = np.linalg.inv(R)
    J = 2.0 * Teff * np.real((D.conj().T @ P_perp @ D) * (A.conj().T @ Rinv @ A).T)
    return 0.5 * (J + J.T)                           # symmetrize


def make_pcrb():
    M, rho = 4, 2
    M_v = rho * (M - 1) + 1                           # = 7 ; virtual limit K = 6
    Kvals = [2, 3, 4, 5, 6, 7, 8]
    sigma2, Teff = 0.5, 80.0
    priors_deg = [0.5, 1.0, 2.0]
    BIG = 1e6

    open_v, closed = [], {s: [] for s in priors_deg}
    for K in Kvals:
        th = np.radians(np.linspace(-50.0, 50.0, K))
        J = _data_fim(th, M_v, sigma2, Teff)
        lam = np.linalg.eigvalsh(J)
        lmin = lam[0]
        open_v.append(min(1.0 / lmin * DEG2, BIG) if lmin > 1e-9 else BIG)
        for s in priors_deg:
            Jp = J + np.eye(K) / np.radians(s) ** 2  # angle prior information
            closed[s].append(1.0 / np.linalg.eigvalsh(Jp)[0] * DEG2)

    fig, ax = plt.subplots(figsize=(3.45, 2.9))
    ax.semilogy(Kvals, open_v, "-s", color="k", lw=1.8, ms=5,
                label="open-loop (data only)")
    cols = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for s, c in zip(priors_deg, cols):
        ax.semilogy(Kvals, closed[s], "--o", color=c, lw=1.5, ms=4,
                    label=fr"closed-loop, $\sigma_\theta={s:.1f}^\circ$")
    ax.axvline(M_v - 1, ls=":", color="#d62728", lw=1.4)
    ax.text(M_v - 1 - 0.1, 4e4, fr"virtual limit $K{{=}}M_v{{-}}1{{=}}{M_v-1}$",
            color="#d62728", fontsize=7.2, rotation=90, va="top", ha="right")
    ax.set_xlabel(r"number of sources $K$  (fixed field of view)")
    ax.set_ylabel(r"worst-case angle variance [deg$^2$]")
    ax.set_ylim(3e-2, 3e6)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=6.8, loc="center left", labelspacing=0.3)
    fig.tight_layout(pad=0.3)
    out = os.path.join(HERE, "fig_pcrb_validation.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("saved", out)


# ---------------------------------------------------------------------------
# Theorem 2: T-COP refinement is minimum-variance; gate enforces benefit
# ---------------------------------------------------------------------------
def make_tcop_mse():
    rng = np.random.RandomState(12345)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.45, 4.85))

    # (a) unbiased prior: ratio = 1/(1+kappa), kappa = sigma_e^2/sigma_p^2
    kap = np.logspace(-2, 2, 200)
    a1.plot(kap, 1.0 / (1.0 + kap), "-", color="#ff7f0e", lw=1.8,
            label=r"closed form $1/(1+\kappa)$")
    kap_mc = np.logspace(-2, 2, 22)
    se2 = 1.0
    ratio_mc = []
    for k in kap_mc:
        sp2 = se2 / k
        xe = rng.normal(0, np.sqrt(se2), 6000)
        xp = rng.normal(0, np.sqrt(sp2), 6000)
        fused = (xe / se2 + xp / sp2) / (1.0 / se2 + 1.0 / sp2)
        ratio_mc.append(np.mean(fused ** 2) / se2)
    a1.plot(kap_mc, ratio_mc, "o", color="#1f77b4", ms=4, label="Monte-Carlo")
    a1.axvline(1.0, ls=":", color="#d62728", lw=1.3)
    a1.text(1.35, 0.63, "prior = data\n(MSE halved)", color="#d62728",
            fontsize=7.0, va="center")
    a1.set_xscale("log")
    a1.set_xlabel(r"prior quality $\kappa=\sigma_e^2/\sigma_p^2$")
    a1.set_ylabel("fused / data-only MSE")
    a1.set_title("(a) unbiased prior: always improves", fontsize=8.5)
    a1.grid(True, which="both", alpha=0.3)
    a1.legend(fontsize=7.0, loc="upper right")

    # (b) biased prior: break-even at beta = ||b||^2 / [D (sigma_e^2+sigma_p^2)] = 1
    beta = np.linspace(0.0, 3.0, 200)
    wp = 0.5                                          # se2 = sp2 = 1 -> prior weight 0.5
    fused_cf = wp * wp * (2.0 * beta) + (1.0 - wp)    # = 0.5*beta + 0.5
    a2.plot(beta, fused_cf, "-", color="#ff7f0e", lw=1.8, label="fused (closed form)")
    beta_mc = np.linspace(0.0, 3.0, 25)
    fused_mc = []
    for b2n in beta_mc:
        b = np.sqrt(2.0 * b2n)                        # ||b||^2 = beta * D *(se2+sp2)=2*beta
        xe = rng.normal(0, 1.0, 6000)
        xp = rng.normal(b, 1.0, 6000)
        fused = 0.5 * xe + 0.5 * xp
        fused_mc.append(np.mean(fused ** 2))
    a2.plot(beta_mc, fused_mc, "o", color="#1f77b4", ms=4, label="fused (MC)")
    # deployed gate: rejects once the Theorem-2 benefit condition is violated, so
    # the estimator clamps to the data-only MSE exactly at the break-even beta=1
    gated = np.minimum(fused_cf, 1.0)
    a2.plot(beta, gated, "-.", color="#2ca02c", lw=1.8, label="gated (deployed)")
    a2.axhline(1.0, ls=":", color="k", lw=1.0)
    a2.axvline(1.0, ls=":", color="#d62728", lw=1.3)
    a2.text(1.06, 1.7, r"break-even" "\n" r"$\|b\|^2{=}D(\sigma_e^2{+}\sigma_p^2)$",
            color="#d62728", fontsize=7.0, va="center")
    a2.set_xlabel(r"normalized squared bias $\|b\|^2/[D(\sigma_e^2+\sigma_p^2)]$")
    a2.set_ylabel("MSE / data-only MSE")
    a2.set_title("(b) biased prior: gate enforces no-harm", fontsize=8.5)
    a2.grid(True, alpha=0.3)
    a2.legend(fontsize=7.0, loc="upper left")

    fig.tight_layout(pad=0.4)
    out = os.path.join(HERE, "fig_tcop_mse_validation.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    make_pcrb()
    make_tcop_mse()
