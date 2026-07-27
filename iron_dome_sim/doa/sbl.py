"""Multisnapshot Sparse Bayesian Learning (SBL) for DOA estimation.

Genuine SBL (Type-II maximum-likelihood / automatic relevance determination),
NOT an l1 sparse-recovery surrogate. This is the multisnapshot SBL of the
Wipf--Rao / Gerstoft et al. line:

    x_l = A_grid s_l + n_l ,   s_{g,l} ~ CN(0, gamma_g) ,   n_l ~ CN(0, sigma2 I)

with a grid of candidate DOAs. The per-grid-point powers gamma_g (hyperparameters)
are learned by evidence maximization; sparsity is induced automatically (most
gamma_g -> 0). Because it is grid/dictionary based it handles the underdetermined
regime K > M-1 directly. Implements the same DOAEstimator interface as MUSIC/COP
so it drops into the eval_mc harness (SBL front-end -> TO-PHD back-end).

Reference: P. Gerstoft, C. F. Mecklenbraeuker, A. Xenaki, S. Nannuru,
"Multisnapshot Sparse Bayesian Learning for DOA," IEEE SPL 23(10):1469-1473, 2016
(doi:10.1109/LSP.2016.2598550); D. P. Wipf and B. D. Rao, IEEE TSP 52(8), 2004.
"""

import numpy as np
from .base import DOAEstimator
from .spectrum import find_peaks_doa


class SBL(DOAEstimator):
    def __init__(self, array, num_sources=None, grid_deg=1.0, n_iter=100,
                 tol=1e-4, fov_deg=85.0, sigma_frac=0.25, power=0.5):
        super().__init__(array, num_sources)
        self.name = "SBL"
        # dictionary grid over the field of view (endfire excluded: the ULA
        # manifold is degenerate near +-90 deg and traps spurious peaks)
        self.grid = np.radians(np.arange(-fov_deg, fov_deg + 1e-9, grid_deg))
        self.dtheta = self.grid[1] - self.grid[0]
        self.n_iter = n_iter
        self.tol = tol
        self.sigma_frac = sigma_frac      # sigma^2 = sigma_frac * lambda_min(R)
        self.power = power                # gamma <- gamma * (num/den)^power

    @property
    def is_underdetermined(self):
        return True

    @property
    def max_sources(self):
        return len(self.grid) - 1

    # ---- SBL core: Gerstoft-style multiplicative evidence-maximization ----
    # gamma_g <- gamma_g * [ (a_g^H S^{-1} R S^{-1} a_g) / (a_g^H S^{-1} a_g) ]^power,
    # S = sigma^2 I + A diag(gamma) A^H. Stable and sparsity-inducing; sigma^2 is a
    # fixed noise estimate (a fraction of lambda_min(R)).
    def _run_sbl(self, X):
        M, L = X.shape
        A = self.array.steering_matrix(self.grid)          # M x G
        Ry = (X @ X.conj().T) / L                           # M x M sample covariance
        lam_min = float(np.linalg.eigvalsh(Ry).real[0])
        sigma2 = max(self.sigma_frac * lam_min, 1e-9)
        gamma = np.real(np.einsum('mg,mn,ng->g', A.conj(), Ry, A)) / (M * M)  # matched filter
        gamma = np.maximum(gamma, 1e-12)
        I = np.eye(M)
        for _ in range(self.n_iter):
            Sigma = sigma2 * I + (A * gamma) @ A.conj().T
            Sinv = np.linalg.inv(Sigma)
            den = np.real(np.einsum('mg,mg->g', A.conj(), Sinv @ A))          # a^H S^{-1} a
            SRS = Sinv @ Ry @ Sinv
            num = np.real(np.einsum('mg,mn,ng->g', A.conj(), SRS, A))         # a^H S^{-1} R S^{-1} a
            ratio = np.maximum(num, 0.0) / np.maximum(den, 1e-12)
            gamma_new = gamma * (ratio ** self.power)
            delta = np.max(np.abs(gamma_new - gamma))
            gamma = gamma_new
            if delta < self.tol * max(np.max(gamma), 1e-12):
                break
        return gamma, sigma2

    def _peaks(self, gamma, K):
        g = gamma
        loc = [i for i in range(1, len(g) - 1) if g[i] > g[i - 1] and g[i] >= g[i + 1]]
        if len(loc) < K:
            loc = list(np.argsort(g)[::-1][:max(K, len(loc))])
        loc = sorted(set(loc), key=lambda i: g[i], reverse=True)[:K]
        out = []
        for i in loc:
            if 0 < i < len(g) - 1:                             # parabolic sub-grid refinement
                y0, y1, y2 = g[i - 1], g[i], g[i + 1]
                den = (y0 - 2 * y1 + y2)
                d = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
                out.append(self.grid[i] + np.clip(d, -1, 1) * self.dtheta)
            else:
                out.append(self.grid[i])
        return np.array(sorted(out))

    def estimate(self, X, scan_angles=None):
        if scan_angles is None:
            scan_angles = self._default_scan_angles()
        gamma, _ = self._run_sbl(X)
        K = self.num_sources if self.num_sources is not None else max(self._estimate_num_sources(X), 1)
        doas = self._peaks(gamma, K)
        P = np.interp(scan_angles, self.grid, gamma)           # align spectrum to scan grid
        pmax = P.max()
        if pmax > 0:
            P = P / pmax
        return doas, P

    def spectrum(self, X, scan_angles):
        gamma, _ = self._run_sbl(X)
        P = np.interp(scan_angles, self.grid, gamma)
        pmax = P.max()
        return P / pmax if pmax > 0 else P
