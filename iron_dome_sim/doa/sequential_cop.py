"""Sequential Deflation COP (SD-COP): Iterative source subtraction and re-estimation.

Novel extension that overcomes the limitation of standard COP when the
number of sources approaches the virtual array capacity.

Key innovation:
1. Estimate strongest sources first via COP
2. Subtract their contribution from the received signal
3. Re-compute cumulant on residual signal
4. Repeat COP on residual for remaining weaker sources
5. Combine all estimates

This is analogous to SIC (Successive Interference Cancellation) in
communications, but applied to higher-order cumulant DOA estimation.

Patent-relevant novel contributions:
- Sequential deflation in the higher-order cumulant domain
- Signal reconstruction and subtraction using estimated DOA + beamforming
- Progressive subspace refinement across deflation stages
- Automatic termination based on residual energy ratio

Reference (base algorithm):
    Choi & Yoo, IEEE TSP 2015.
"""

import numpy as np
from .base import DOAEstimator
from .spectrum import find_peaks_doa
from ..signal_model.cumulant import compute_cumulant_matrix


class SequentialDeflationCOP(DOAEstimator):
    """Sequential Deflation COP for enhanced underdetermined DOA estimation.

    Addresses the capacity limitation by iteratively:
    1. Applying COP to estimate a batch of DOAs
    2. Reconstructing and subtracting the estimated source signals
    3. Re-applying COP to the residual for remaining sources

    This effectively extends the practical resolution beyond
    rho*(M-1) sources by processing in stages.

    Args:
        array: ULA array object.
        rho: Cumulant order parameter (default: 2).
        num_sources: Total number of sources to estimate.
        batch_size: Sources to estimate per deflation stage.
                    Default: M-1 (maximum for stable COP per stage).
        max_stages: Maximum number of deflation stages.
        residual_threshold: Stop when residual energy ratio drops below this.
        spectrum_type: COP spectrum type ("combined", "signal", "noise").
    """

    def __init__(self, array, rho=2, num_sources=None, batch_size=None,
                 max_stages=5, residual_threshold=0.01,
                 spectrum_type="combined"):
        super().__init__(array, num_sources)
        self.rho = rho
        self.batch_size = batch_size or max(array.max_sources(rho) // 2, 3)
        self.max_stages = max_stages
        self.residual_threshold = residual_threshold
        self.spectrum_type = spectrum_type
        self.name = f"SD-COP-{2*rho}th"

        # Diagnostics
        self.stage_results = []

    @property
    def is_underdetermined(self):
        return True

    @property
    def max_sources(self):
        # Exceeds single-stage COP limit via deflation
        return self.array.max_sources(self.rho) * self.max_stages

    def estimate(self, X, scan_angles=None):
        """Estimate DOAs via sequential deflation.

        Strategy:
        - If K is within single-stage COP capacity: use standard COP (no deflation)
        - If K exceeds capacity: deflate in stages, then globally refine

        Args:
            X: Received signal matrix, shape (M, T).
            scan_angles: Angular scan grid in radians.

        Returns:
            doa_estimates: All estimated DOA angles in radians.
            spectrum: Combined spectrum from all stages.
        """
        if scan_angles is None:
            scan_angles = self._default_scan_angles()

        M, T = X.shape
        self.stage_results = []

        K_total = self.num_sources
        if K_total is None:
            K_total = self._estimate_num_sources(X)

        single_stage_capacity = self.array.max_sources(self.rho)

        # If within single-stage capacity, use standard COP (no deflation)
        if K_total <= single_stage_capacity:
            doas, spectrum = self._cop_stage(X, scan_angles, K_total)
            self.stage_results.append({
                'stage': 0, 'doas': doas,
                'energy_ratio': 1.0, 'n_detected': len(doas),
            })
            return doas, spectrum

        # Beyond capacity: sequential deflation
        X_residual = X.copy()
        all_doas = []
        combined_spectrum = np.zeros(len(scan_angles))
        original_energy = np.linalg.norm(X) ** 2
        K_remaining = K_total

        for stage in range(self.max_stages):
            if K_remaining <= 0:
                break

            # Check residual energy
            residual_energy = np.linalg.norm(X_residual) ** 2
            energy_ratio = residual_energy / (original_energy + 1e-30)

            if energy_ratio < self.residual_threshold and stage > 0:
                break

            # Stage batch size: estimate up to capacity per stage
            K_batch = min(self.batch_size, K_remaining, single_stage_capacity)

            # Apply COP to residual
            stage_doas, stage_spectrum = self._cop_stage(
                X_residual, scan_angles, K_batch
            )

            if len(stage_doas) == 0:
                break

            # Record stage results
            self.stage_results.append({
                'stage': stage,
                'doas': stage_doas,
                'energy_ratio': energy_ratio,
                'n_detected': len(stage_doas),
            })

            # Accumulate results
            all_doas.extend(stage_doas)
            combined_spectrum = np.maximum(combined_spectrum, stage_spectrum)
            K_remaining -= len(stage_doas)

            # Deflation: subtract ALL found sources from ORIGINAL signal
            # (cumulative deflation from original avoids error accumulation)
            X_residual = self._deflate(X, np.array(all_doas))

        # Remove duplicate DOAs
        all_doas = self._remove_duplicates(np.array(all_doas))

        # Global refinement: re-estimate DOAs using all found sources
        all_doas = self._global_refinement(X, all_doas, scan_angles)

        # Normalize combined spectrum
        P_max = np.max(combined_spectrum)
        if P_max > 0:
            combined_spectrum /= P_max

        return all_doas, combined_spectrum

    def spectrum(self, X, scan_angles):
        """Compute combined spectrum from all deflation stages."""
        _, combined = self.estimate(X, scan_angles)
        return combined

    def _cop_stage(self, X, scan_angles, K):
        """Single COP estimation stage on (possibly deflated) data.

        Args:
            X: Signal matrix (original or residual).
            scan_angles: Angular grid.
            K: Number of sources to find in this stage.

        Returns:
            doas: Estimated DOAs for this stage.
            spectrum: Spatial spectrum for this stage.
        """
        M_v_target = self.array.virtual_array_size(self.rho)

        # Compute cumulant of residual
        C = compute_cumulant_matrix(X, self.rho)
        M_v = C.shape[0]

        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(C)
        idx = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        K = min(K, M_v - 1)
        U_s = eigenvectors[:, :K]
        U_n = eigenvectors[:, K:]

        # COP spectrum
        P = np.zeros(len(scan_angles))
        for i, theta in enumerate(scan_angles):
            a_v = self.array.virtual_steering_vector(theta, self.rho)
            if len(a_v) != M_v:
                a_v = a_v[:M_v] if len(a_v) > M_v else np.pad(a_v, (0, M_v - len(a_v)))

            sig_proj = U_s.conj().T @ a_v
            numerator = np.real(np.sum(np.abs(sig_proj) ** 2))

            noise_proj = U_n.conj().T @ a_v
            denominator = np.real(np.sum(np.abs(noise_proj) ** 2))

            if denominator < 1e-15:
                P[i] = 1e10 * (numerator + 1e-15)
            else:
                P[i] = numerator / denominator

        P_max = np.max(P)
        if P_max > 0:
            P /= P_max

        doas = find_peaks_doa(P, scan_angles, K)
        return doas, P

    def _deflate(self, X, estimated_doas):
        """Subtract estimated source contributions from received signal.

        Uses least-squares estimation of source signals given the estimated
        DOAs, then subtracts the reconstructed signal.

        This is more accurate than per-source MVDR beamforming because it
        jointly estimates all sources, avoiding cross-source leakage.

        Args:
            X: Current signal matrix, shape (M, T).
            estimated_doas: DOAs to subtract (radians).

        Returns:
            X_residual: Signal after source subtraction.
        """
        M, T = X.shape

        if len(estimated_doas) == 0:
            return X

        # Steering matrix for estimated DOAs
        A_est = self.array.steering_matrix(estimated_doas)

        # Joint least-squares source estimation: S = (A^H A)^{-1} A^H X
        AHA = A_est.conj().T @ A_est
        AHA_reg = AHA + 1e-6 * np.eye(len(estimated_doas))
        S_est = np.linalg.solve(AHA_reg, A_est.conj().T @ X)

        # Reconstruct and subtract
        X_reconstructed = A_est @ S_est
        X_residual = X - X_reconstructed

        return X_residual

    def _global_refinement(self, X, doas, scan_angles, search_range_deg=3.0):
        """Globally refine all DOA estimates after deflation stages.

        For each estimated DOA, performs a fine local search on the original
        signal data with all OTHER sources deflated. This corrects errors
        from inter-stage deflation artifacts.

        Novel contribution: Post-deflation global refinement in higher-order
        cumulant domain.

        Args:
            X: Original received signal matrix, shape (M, T).
            doas: All estimated DOAs from deflation stages.
            scan_angles: Angular grid.
            search_range_deg: Local search range around each DOA (degrees).

        Returns:
            refined_doas: Refined DOA estimates.
        """
        if len(doas) <= 1:
            return doas

        search_range = np.radians(search_range_deg)
        refined = np.array(doas).copy()

        for k in range(len(refined)):
            # Deflate all sources EXCEPT source k
            other_doas = np.concatenate([refined[:k], refined[k+1:]])
            X_k = self._deflate(X, other_doas)

            # Compute COP spectrum on residual (only source k remains)
            C = compute_cumulant_matrix(X_k, self.rho)
            M_v = C.shape[0]

            eigenvalues, eigenvectors = np.linalg.eigh(C)
            idx = np.argsort(np.abs(eigenvalues))[::-1]
            eigenvectors = eigenvectors[:, idx]

            U_s = eigenvectors[:, :1]   # Single source
            U_n = eigenvectors[:, 1:]

            # Local search around current estimate
            local_angles = scan_angles[
                (scan_angles > refined[k] - search_range) &
                (scan_angles < refined[k] + search_range)
            ]

            if len(local_angles) == 0:
                continue

            best_val = -1
            best_angle = refined[k]

            for theta in local_angles:
                a_v = self.array.virtual_steering_vector(theta, self.rho)
                if len(a_v) != M_v:
                    a_v = a_v[:M_v] if len(a_v) > M_v else np.pad(
                        a_v, (0, M_v - len(a_v)))

                sig_proj = U_s.conj().T @ a_v
                numerator = np.real(np.sum(np.abs(sig_proj) ** 2))
                noise_proj = U_n.conj().T @ a_v
                denominator = np.real(np.sum(np.abs(noise_proj) ** 2))

                val = numerator / (denominator + 1e-15)
                if val > best_val:
                    best_val = val
                    best_angle = theta

            refined[k] = best_angle

        return np.sort(refined)

    def _remove_duplicates(self, doas, threshold_rad=0.02):
        """Remove duplicate DOA estimates (within threshold)."""
        if len(doas) <= 1:
            return doas

        doas_sorted = np.sort(doas)
        unique = [doas_sorted[0]]

        for d in doas_sorted[1:]:
            if abs(d - unique[-1]) > threshold_rad:
                unique.append(d)

        return np.array(unique)
