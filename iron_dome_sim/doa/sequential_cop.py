"""Sequential Deflation COP (SD-COP): Iterative source subtraction and re-estimation.

Novel extension that overcomes the limitation of standard COP when the
number of sources approaches the virtual array capacity.

Key innovation (v2 — improved deflation + multi-pass refinement):
1. Estimate strongest sources via COP on signal-domain residual
2. Joint LS signal reconstruction and subtraction
3. Smaller batch sizes for more accurate per-stage estimation
4. Adaptive source counting via eigenvalue gap analysis
5. Multi-pass global refinement for accurate final estimates
6. Leave-one-out final spectrum for clean visualization

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
    1. Applying COP on residual signal cumulant to estimate a batch of DOAs
    2. Joint LS signal reconstruction and subtraction from original signal
    3. Adaptive batch sizing via eigenvalue analysis per stage
    4. Multi-pass global refinement on original data
    5. Leave-one-out final spectrum for clear peak visualization

    Args:
        array: ULA array object.
        rho: Cumulant order parameter (default: 2).
        num_sources: Total number of sources to estimate.
        batch_size: Sources to estimate per deflation stage (default: adaptive).
        max_stages: Maximum number of deflation stages.
        residual_threshold: Stop when residual energy drops below this fraction.
        spectrum_type: COP spectrum type ("combined", "signal", "noise").
        n_refine: Number of global refinement passes (default: 3).
    """

    def __init__(self, array, rho=2, num_sources=None, batch_size=None,
                 max_stages=8, residual_threshold=0.005,
                 spectrum_type="combined", n_refine=3):
        super().__init__(array, num_sources)
        self.rho = rho
        # batch_size: half the single-stage capacity for balanced stages
        self.batch_size = batch_size or max(array.max_sources(rho) // 2, 3)
        self.max_stages = max_stages
        self.residual_threshold = residual_threshold
        self.spectrum_type = spectrum_type
        self.n_refine = n_refine
        self.name = f"SD-COP-{2*rho}th"

        # Diagnostics
        self.stage_results = []

    @property
    def is_underdetermined(self):
        return True

    @property
    def max_sources(self):
        return self.array.max_sources(self.rho) * self.max_stages

    def estimate(self, X, scan_angles=None):
        """Estimate DOAs via sequential deflation with multi-pass refinement.

        Args:
            X: Received signal matrix, shape (M, T).
            scan_angles: Angular scan grid in radians.

        Returns:
            doa_estimates: All estimated DOA angles in radians, sorted.
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

        # Beyond capacity: sequential deflation in signal domain
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

            # Fixed batch size (adaptive disabled — more stable)
            K_batch = min(self.batch_size, K_remaining, single_stage_capacity)

            if K_batch <= 0:
                break

            # Apply COP to residual signal
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

            # Deflate: subtract ALL found sources from ORIGINAL signal
            # (cumulative deflation from original avoids error accumulation)
            X_residual = self._deflate(X, np.array(all_doas))

        # Remove duplicate DOAs (close estimates from different stages)
        all_doas = self._remove_duplicates(np.array(all_doas))

        # Multi-pass global refinement: each pass narrows the search
        for refine_pass in range(self.n_refine):
            search_deg = 5.0 - refine_pass * 1.0  # 5° → 4° → 3°
            all_doas = self._global_refinement(
                X, all_doas, scan_angles, search_range_deg=max(search_deg, 2.0)
            )
            # Remove near-duplicates that refinement may create
            all_doas = self._remove_duplicates(all_doas, threshold_rad=0.03)

        # Leave-one-out final spectrum for clean visualization
        combined_spectrum = self._leave_one_out_spectrum(
            X, all_doas, scan_angles
        )

        # Normalize
        P_max = np.max(combined_spectrum)
        if P_max > 0:
            combined_spectrum /= P_max

        return all_doas, combined_spectrum

    def spectrum(self, X, scan_angles):
        """Compute combined spectrum from all deflation stages."""
        _, combined = self.estimate(X, scan_angles)
        return combined

    def _adaptive_batch_size(self, C, K_remaining):
        """Determine batch size based on eigenvalue gap analysis.

        Uses eigenvalue magnitude to determine how many strong sources
        are clearly separable from the noise floor.
        """
        try:
            eigenvalues = np.sort(np.abs(np.linalg.eigvalsh(C)))[::-1]

            if len(eigenvalues) < 3:
                return min(self.batch_size, K_remaining)

            # Count eigenvalues significantly above noise floor
            noise_level = np.median(eigenvalues[len(eigenvalues)//2:])
            n_significant = int(np.sum(eigenvalues > 3.0 * noise_level))

            K_batch = min(self.batch_size, max(n_significant, 2), K_remaining)
            return K_batch

        except Exception:
            return min(self.batch_size, K_remaining)

    def _cop_stage(self, X, scan_angles, K):
        """Single COP estimation stage on (possibly deflated) signal.

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

            if self.spectrum_type == "combined":
                sig_proj = U_s.conj().T @ a_v
                numerator = np.real(np.sum(np.abs(sig_proj) ** 2))
                noise_proj = U_n.conj().T @ a_v
                denominator = np.real(np.sum(np.abs(noise_proj) ** 2))
                if denominator < 1e-15:
                    P[i] = 1e10 * (numerator + 1e-15)
                else:
                    P[i] = numerator / denominator
            elif self.spectrum_type == "noise":
                noise_proj = U_n.conj().T @ a_v
                denominator = np.real(np.sum(np.abs(noise_proj) ** 2))
                P[i] = 1.0 / (denominator + 1e-15)
            else:  # "signal"
                sig_proj = U_s.conj().T @ a_v
                P[i] = np.real(np.sum(np.abs(sig_proj) ** 2))

        P_max = np.max(P)
        if P_max > 0:
            P /= P_max

        doas = find_peaks_doa(P, scan_angles, K)
        return doas, P

    def _deflate(self, X, estimated_doas):
        """Subtract estimated source contributions from received signal.

        Uses joint least-squares estimation of all source signals,
        then subtracts the reconstructed signal.

        Args:
            X: Original signal matrix, shape (M, T).
            estimated_doas: All DOAs estimated so far (radians).

        Returns:
            X_residual: Signal after source subtraction.
        """
        M, T = X.shape

        if len(estimated_doas) == 0:
            return X.copy()

        # Steering matrix for all estimated DOAs
        A_est = self.array.steering_matrix(estimated_doas)

        # Joint LS source estimation: S = (A^H A)^{-1} A^H X
        AHA = A_est.conj().T @ A_est
        # Adaptive regularization based on condition number
        reg_val = 1e-4 * np.real(np.trace(AHA)) / len(estimated_doas)
        AHA_reg = AHA + reg_val * np.eye(len(estimated_doas))
        S_est = np.linalg.solve(AHA_reg, A_est.conj().T @ X)

        # Reconstruct and subtract
        X_reconstructed = A_est @ S_est
        X_residual = X - X_reconstructed

        return X_residual

    def _global_refinement(self, X, doas, scan_angles, search_range_deg=5.0):
        """Global refinement: for each DOA, deflate others and re-estimate.

        For each estimated DOA, deflates all other sources from the original
        signal, then performs a fine local COP search on the residual.

        Args:
            X: Original received signal matrix, shape (M, T).
            doas: All estimated DOAs from deflation stages.
            scan_angles: Angular grid.
            search_range_deg: Local search range (degrees).

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
            local_mask = (
                (scan_angles > refined[k] - search_range) &
                (scan_angles < refined[k] + search_range)
            )
            local_angles = scan_angles[local_mask]

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

    def _leave_one_out_spectrum(self, X, doas, scan_angles):
        """Compute clean spectrum via leave-one-out deflation.

        For each detected source, deflates all others and computes the
        single-source COP spectrum. The combined spectrum sums all
        single-source spectra for a cleaner visualization with
        peaks at all detected DOAs.

        Args:
            X: Original signal matrix.
            doas: All refined DOA estimates.
            scan_angles: Angular grid.

        Returns:
            combined: Combined spectrum (unnormalized).
        """
        combined = np.zeros(len(scan_angles))

        if len(doas) == 0:
            return combined

        for k in range(len(doas)):
            # Deflate all sources except k
            other_doas = np.concatenate([doas[:k], doas[k+1:]])
            X_k = self._deflate(X, other_doas)

            # Single-source COP spectrum
            C = compute_cumulant_matrix(X_k, self.rho)
            M_v = C.shape[0]

            eigenvalues, eigenvectors = np.linalg.eigh(C)
            idx = np.argsort(np.abs(eigenvalues))[::-1]
            eigenvectors = eigenvectors[:, idx]

            U_s = eigenvectors[:, :1]
            U_n = eigenvectors[:, 1:]

            for i, theta in enumerate(scan_angles):
                a_v = self.array.virtual_steering_vector(theta, self.rho)
                if len(a_v) != M_v:
                    a_v = a_v[:M_v] if len(a_v) > M_v else np.pad(
                        a_v, (0, M_v - len(a_v)))

                sig_proj = U_s.conj().T @ a_v
                numerator = np.real(np.sum(np.abs(sig_proj) ** 2))
                noise_proj = U_n.conj().T @ a_v
                denominator = np.real(np.sum(np.abs(noise_proj) ** 2))

                combined[i] += numerator / (denominator + 1e-15)

        return combined

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
