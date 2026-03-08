# Iron Dome DOA Estimation + Tracking Simulation

Underdetermined high-resolution DOA estimation using **2ρth-order Subspace Constrained Optimization (COP)**, extended with multi-target tracking for missile defense simulation.

## Key Innovation

**Fewer sensors, more targets**: With M=8 sensors, the proposed COP algorithm can resolve up to **2ρ(M-1) = 28 sources** (at ρ=2), far exceeding the M-1=7 limit of classical methods (MUSIC, ESPRIT).

## Based on

> Choi, J.H. and Yoo, C.D., "Underdetermined High-Resolution DOA Estimation: A 2ρth-Order Source-Signal/Noise Subspace Constrained Optimization," *IEEE Trans. Signal Processing*, vol. 63, no. 7, pp. 1858-1873, 2015.

## Quick Start

```bash
pip install numpy scipy matplotlib
python examples/demo_iron_dome.py
```

## Project Structure

```
iron_dome_sim/
├── signal_model/    # Array geometry, signal generation, higher-order cumulants
├── doa/             # DOA estimation: COP (proposed), MUSIC, ESPRIT, Capon, L1-SVD, LASSO
├── tracking/        # Multi-target tracking: EKF/UKF/PF, GNN/JPDA, track management
├── scenario/        # Iron Dome scenarios: threats, radar network, interception
├── eval/            # Metrics (RMSE, GOSPA, CRLB) and Monte Carlo evaluation
└── viz/             # 3D visualization, spectrum plots, animation
```

## Algorithms Compared

| Algorithm | Type | Underdetermined | Max Sources (M=8) |
|-----------|------|-----------------|-------------------|
| **COP-4th (Proposed)** | 4th-order subspace | Yes | 28 |
| MUSIC | 2nd-order subspace | No | 7 |
| ESPRIT | 2nd-order subspace | No | 7 |
| Capon/MVDR | Beamforming | No | Limited |
| L1-SVD | Sparse recovery | Yes | Grid-dependent |
| LASSO | Sparse recovery | Yes | Grid-dependent |
