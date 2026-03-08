"""Abstract base class for DOA estimation algorithms."""

from abc import ABC, abstractmethod
import numpy as np


class DOAEstimator(ABC):
    """Base class for all DOA estimation algorithms.

    Subclasses must implement estimate() which returns estimated DOA angles.
    """

    def __init__(self, array, num_sources=None):
        """
        Args:
            array: Array geometry object (ULA or URA).
            num_sources: Number of sources (None = unknown, estimate it).
        """
        self.array = array
        self.num_sources = num_sources
        self.name = self.__class__.__name__

    @abstractmethod
    def estimate(self, X, scan_angles=None):
        """Estimate DOA angles from received data.

        Args:
            X: Received signal matrix, shape (M, T).
            scan_angles: Angular grid for spectrum search, in radians.
                         Default: -90 to 90 degrees, 0.1 degree step.

        Returns:
            doa_estimates: Estimated DOA angles in radians.
            spectrum: Spatial spectrum (if applicable), or None.
        """
        pass

    @abstractmethod
    def spectrum(self, X, scan_angles):
        """Compute spatial spectrum over scan angles.

        Args:
            X: Received signal matrix, shape (M, T).
            scan_angles: Angular grid in radians.

        Returns:
            P: Spatial spectrum values, shape matching scan_angles.
        """
        pass

    def _default_scan_angles(self):
        """Default angular scan grid: -90 to 90 deg, 0.1 deg step."""
        return np.linspace(-np.pi / 2, np.pi / 2, 1801)

    def _estimate_num_sources(self, X):
        """Estimate number of sources using MDL or AIC.

        Uses the Minimum Description Length criterion.

        Args:
            X: Received signal matrix, shape (M, T).

        Returns:
            Estimated number of sources.
        """
        M, T = X.shape
        R = X @ X.conj().T / T
        eigenvalues = np.sort(np.real(np.linalg.eigvalsh(R)))[::-1]

        # MDL criterion
        mdl_values = []
        for k in range(M):
            # Log-likelihood ratio
            noise_eigs = eigenvalues[k:]
            geo_mean = np.exp(np.mean(np.log(noise_eigs + 1e-30)))
            ari_mean = np.mean(noise_eigs)

            log_likelihood = -(M - k) * T * np.log(geo_mean / (ari_mean + 1e-30))

            # Penalty term
            penalty = 0.5 * k * (2 * M - k) * np.log(T)

            mdl_values.append(log_likelihood + penalty)

        return np.argmin(mdl_values)

    @property
    def is_underdetermined(self):
        """Whether this algorithm can handle K > M (underdetermined) case."""
        return False

    @property
    def max_sources(self):
        """Maximum number of sources this algorithm can resolve."""
        return self.array.M - 1
