"""Array geometry definitions for radar sensor arrays.

Supports Uniform Linear Array (ULA) and Uniform Rectangular Array (URA)
with steering vector computation for standard and virtual (higher-order) arrays.
"""

import numpy as np


class UniformLinearArray:
    """Uniform Linear Array (ULA) with M sensors spaced d apart.

    Args:
        M: Number of sensors.
        d: Inter-element spacing in wavelengths (default: 0.5).
    """

    def __init__(self, M, d=0.5):
        self.M = M
        self.d = d
        self.positions = np.arange(M) * d  # sensor positions in wavelengths

    def steering_vector(self, theta):
        """Compute steering vector for azimuth angle theta.

        Args:
            theta: Azimuth angle in radians.

        Returns:
            Complex steering vector of shape (M,).
        """
        return np.exp(1j * 2 * np.pi * self.d * np.arange(self.M) * np.sin(theta))

    def steering_matrix(self, thetas):
        """Compute steering matrix for multiple angles.

        Args:
            thetas: Array of azimuth angles in radians, shape (K,).

        Returns:
            Steering matrix of shape (M, K).
        """
        return np.column_stack([self.steering_vector(t) for t in thetas])

    def virtual_steering_vector(self, theta, rho):
        """Compute virtual steering vector for 2rho-th order processing.

        The sum co-array from the rho-fold Kronecker product has
        rho*(M-1)+1 unique positions, matching the cumulant matrix dimension.

        Args:
            theta: Azimuth angle in radians.
            rho: Order parameter (rho >= 1).

        Returns:
            Virtual steering vector of shape (rho*(M-1)+1,).
        """
        M_v = rho * (self.M - 1) + 1
        indices = np.arange(M_v)
        return np.exp(1j * 2 * np.pi * self.d * indices * np.sin(theta))

    def virtual_array_size(self, rho):
        """Return virtual array size for given order rho."""
        return rho * (self.M - 1) + 1

    def max_sources(self, rho):
        """Maximum number of resolvable sources for given order rho.

        With the combined signal+noise subspace COP spectrum,
        can resolve up to rho*(M-1) sources.
        """
        return rho * (self.M - 1)


class UniformRectangularArray:
    """Uniform Rectangular Array (URA) for 3D DOA estimation.

    Args:
        Mx: Number of sensors along x-axis.
        My: Number of sensors along y-axis.
        d: Inter-element spacing in wavelengths (default: 0.5).
    """

    def __init__(self, Mx, My, d=0.5):
        self.Mx = Mx
        self.My = My
        self.M = Mx * My
        self.d = d
        # Sensor positions as (x, y) pairs
        px = np.arange(Mx) * d
        py = np.arange(My) * d
        self.positions_x, self.positions_y = np.meshgrid(px, py)
        self.positions_x = self.positions_x.ravel()
        self.positions_y = self.positions_y.ravel()

    def steering_vector(self, theta, phi):
        """Compute steering vector for azimuth theta and elevation phi.

        Args:
            theta: Azimuth angle in radians.
            phi: Elevation angle in radians.

        Returns:
            Complex steering vector of shape (M,).
        """
        ux = np.sin(theta) * np.cos(phi)
        uy = np.sin(phi)
        return np.exp(1j * 2 * np.pi * (self.positions_x * ux + self.positions_y * uy))

    def steering_matrix(self, thetas, phis):
        """Compute steering matrix for multiple (azimuth, elevation) pairs.

        Args:
            thetas: Azimuth angles in radians, shape (K,).
            phis: Elevation angles in radians, shape (K,).

        Returns:
            Steering matrix of shape (M, K).
        """
        return np.column_stack(
            [self.steering_vector(t, p) for t, p in zip(thetas, phis)]
        )
