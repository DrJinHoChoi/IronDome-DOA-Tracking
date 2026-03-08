"""State estimation filters for target tracking.

Implements EKF, UKF, and Particle Filter for nonlinear state estimation
in DOA-based target tracking.
"""

import numpy as np


class ExtendedKalmanFilter:
    """Extended Kalman Filter for nonlinear state estimation.

    Linearizes the nonlinear system model using first-order Taylor expansion.
    Suitable for mildly nonlinear systems.
    """

    def __init__(self, model, x0=None, P0=None):
        """
        Args:
            model: Motion model with F/f, H/h, Q, R methods.
            x0: Initial state estimate.
            P0: Initial covariance.
        """
        self.model = model
        dim = model.dim_state
        self.x = x0 if x0 is not None else np.zeros(dim)
        self.P = P0 if P0 is not None else np.eye(dim) * 10.0

    def predict(self):
        """Time update (prediction step)."""
        # State prediction
        if hasattr(self.model, 'f'):
            self.x = self.model.f(self.x)
        else:
            self.x = self.model.F(self.x) @ self.x

        # Covariance prediction
        F = self.model.F(self.x)
        Q = self.model.Q(self.x)
        self.P = F @ self.P @ F.T + Q

        return self.x.copy(), self.P.copy()

    def update(self, z):
        """Measurement update (correction step).

        Args:
            z: Measurement vector.

        Returns:
            Updated state and covariance.
        """
        # Innovation
        if hasattr(self.model, 'h'):
            z_pred = self.model.h(self.x)
            H = self.model.H(self.x)
        else:
            H = self.model.H(self.x)
            z_pred = H @ self.x

        y = z - z_pred

        # Wrap angle differences to [-pi, pi]
        for i in range(len(y)):
            y[i] = _wrap_angle(y[i])

        # Innovation covariance
        R = self.model.R()
        S = H @ self.P @ H.T + R

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y

        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(self.model.dim_state) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

        return self.x.copy(), self.P.copy()

    def innovation(self, z):
        """Compute innovation and its covariance (for data association)."""
        if hasattr(self.model, 'h'):
            z_pred = self.model.h(self.x)
            H = self.model.H(self.x)
        else:
            H = self.model.H(self.x)
            z_pred = H @ self.x

        y = z - z_pred
        for i in range(len(y)):
            y[i] = _wrap_angle(y[i])

        S = H @ self.P @ H.T + self.model.R()
        return y, S


class UnscentedKalmanFilter:
    """Unscented Kalman Filter for nonlinear state estimation.

    Uses sigma points to capture mean and covariance through nonlinear
    transformations. More accurate than EKF for highly nonlinear systems.
    """

    def __init__(self, model, x0=None, P0=None, alpha=1e-3, beta=2.0, kappa=0.0):
        self.model = model
        dim = model.dim_state
        self.x = x0 if x0 is not None else np.zeros(dim)
        self.P = P0 if P0 is not None else np.eye(dim) * 10.0
        self.n = dim
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self._compute_weights()

    def _compute_weights(self):
        """Compute sigma point weights."""
        n = self.n
        lam = self.alpha ** 2 * (n + self.kappa) - n

        self.Wm = np.full(2 * n + 1, 1.0 / (2 * (n + lam)))
        self.Wc = np.full(2 * n + 1, 1.0 / (2 * (n + lam)))
        self.Wm[0] = lam / (n + lam)
        self.Wc[0] = lam / (n + lam) + (1 - self.alpha ** 2 + self.beta)
        self.gamma = np.sqrt(n + lam)

    def _sigma_points(self, x, P):
        """Generate sigma points."""
        n = self.n
        try:
            sqrt_P = np.linalg.cholesky(P)
        except np.linalg.LinAlgError:
            sqrt_P = np.linalg.cholesky(P + 1e-6 * np.eye(n))

        sigma = np.zeros((2 * n + 1, n))
        sigma[0] = x
        for i in range(n):
            sigma[i + 1] = x + self.gamma * sqrt_P[:, i]
            sigma[n + i + 1] = x - self.gamma * sqrt_P[:, i]
        return sigma

    def predict(self):
        sigma = self._sigma_points(self.x, self.P)

        # Transform through process model
        sigma_pred = np.zeros_like(sigma)
        for i in range(len(sigma)):
            if hasattr(self.model, 'f'):
                sigma_pred[i] = self.model.f(sigma[i])
            else:
                sigma_pred[i] = self.model.F() @ sigma[i]

        # Predicted mean
        self.x = np.sum(self.Wm[:, None] * sigma_pred, axis=0)

        # Predicted covariance
        Q = self.model.Q()
        self.P = Q.copy()
        for i in range(len(sigma_pred)):
            d = sigma_pred[i] - self.x
            self.P += self.Wc[i] * np.outer(d, d)

        return self.x.copy(), self.P.copy()

    def update(self, z):
        sigma = self._sigma_points(self.x, self.P)

        # Transform through observation model
        n_obs = self.model.dim_obs
        Z_sigma = np.zeros((len(sigma), n_obs))
        for i in range(len(sigma)):
            if hasattr(self.model, 'h'):
                Z_sigma[i] = self.model.h(sigma[i])
            else:
                Z_sigma[i] = self.model.H() @ sigma[i]

        # Predicted measurement mean
        z_pred = np.sum(self.Wm[:, None] * Z_sigma, axis=0)

        # Innovation covariance
        R = self.model.R()
        Pzz = R.copy()
        Pxz = np.zeros((self.n, n_obs))

        for i in range(len(sigma)):
            dz = Z_sigma[i] - z_pred
            for j in range(len(dz)):
                dz[j] = _wrap_angle(dz[j])
            dx = sigma[i] - self.x

            Pzz += self.Wc[i] * np.outer(dz, dz)
            Pxz += self.Wc[i] * np.outer(dx, dz)

        # Kalman gain
        K = Pxz @ np.linalg.inv(Pzz)

        # Innovation
        y = z - z_pred
        for i in range(len(y)):
            y[i] = _wrap_angle(y[i])

        self.x = self.x + K @ y
        self.P = self.P - K @ Pzz @ K.T

        return self.x.copy(), self.P.copy()

    def innovation(self, z):
        if hasattr(self.model, 'h'):
            z_pred = self.model.h(self.x)
            H = self.model.H(self.x)
        else:
            H = self.model.H()
            z_pred = H @ self.x

        y = z - z_pred
        for i in range(len(y)):
            y[i] = _wrap_angle(y[i])
        S = H @ self.P @ H.T + self.model.R()
        return y, S


class ParticleFilter:
    """Particle Filter (Sequential Monte Carlo) for nonlinear tracking.

    Handles non-Gaussian distributions and multimodal posteriors.
    More computationally expensive but most general.
    """

    def __init__(self, model, x0=None, P0=None, num_particles=500):
        self.model = model
        dim = model.dim_state
        self.N = num_particles

        x_init = x0 if x0 is not None else np.zeros(dim)
        P_init = P0 if P0 is not None else np.eye(dim) * 10.0

        # Initialize particles from prior
        try:
            chol = np.linalg.cholesky(P_init)
        except np.linalg.LinAlgError:
            chol = np.linalg.cholesky(P_init + 1e-6 * np.eye(dim))

        self.particles = x_init + (np.random.randn(self.N, dim) @ chol.T)
        self.weights = np.ones(self.N) / self.N

        # Cached state estimate
        self.x = x_init.copy()
        self.P = P_init.copy()

    def predict(self):
        Q = self.model.Q()
        try:
            chol_Q = np.linalg.cholesky(Q)
        except np.linalg.LinAlgError:
            chol_Q = np.linalg.cholesky(Q + 1e-10 * np.eye(Q.shape[0]))

        noise = np.random.randn(self.N, self.model.dim_state) @ chol_Q.T

        for i in range(self.N):
            if hasattr(self.model, 'f'):
                self.particles[i] = self.model.f(self.particles[i]) + noise[i]
            else:
                self.particles[i] = self.model.F() @ self.particles[i] + noise[i]

        self._update_estimate()
        return self.x.copy(), self.P.copy()

    def update(self, z):
        R = self.model.R()
        R_inv = np.linalg.inv(R)
        R_det = np.linalg.det(R)

        for i in range(self.N):
            if hasattr(self.model, 'h'):
                z_pred = self.model.h(self.particles[i])
            else:
                z_pred = self.model.H() @ self.particles[i]

            innovation = z - z_pred
            for j in range(len(innovation)):
                innovation[j] = _wrap_angle(innovation[j])

            # Gaussian likelihood
            exponent = -0.5 * innovation @ R_inv @ innovation
            self.weights[i] *= np.exp(exponent)

        # Normalize weights
        total = np.sum(self.weights)
        if total < 1e-30:
            self.weights = np.ones(self.N) / self.N
        else:
            self.weights /= total

        # Resample if effective sample size is too low
        N_eff = 1.0 / np.sum(self.weights ** 2)
        if N_eff < self.N / 2:
            self._systematic_resample()

        self._update_estimate()
        return self.x.copy(), self.P.copy()

    def _systematic_resample(self):
        """Systematic resampling to avoid particle degeneracy."""
        cumsum = np.cumsum(self.weights)
        u = (np.random.random() + np.arange(self.N)) / self.N
        indices = np.searchsorted(cumsum, u)
        indices = np.clip(indices, 0, self.N - 1)

        self.particles = self.particles[indices].copy()
        self.weights = np.ones(self.N) / self.N

    def _update_estimate(self):
        """Compute weighted mean and covariance."""
        self.x = np.sum(self.weights[:, None] * self.particles, axis=0)
        diff = self.particles - self.x
        self.P = np.zeros((self.model.dim_state, self.model.dim_state))
        for i in range(self.N):
            self.P += self.weights[i] * np.outer(diff[i], diff[i])

    def innovation(self, z):
        if hasattr(self.model, 'h'):
            z_pred = self.model.h(self.x)
        else:
            z_pred = self.model.H() @ self.x

        y = z - z_pred
        for i in range(len(y)):
            y[i] = _wrap_angle(y[i])

        if hasattr(self.model, 'H'):
            H = self.model.H(self.x) if hasattr(self.model, 'h') else self.model.H()
            S = H @ self.P @ H.T + self.model.R()
        else:
            S = self.P[:2, :2] + self.model.R()

        return y, S


def _wrap_angle(angle):
    """Wrap angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi
