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

## License

**Dual-licensed** — academic (non-commercial) use is free; commercial use
requires a separate license from the author.

- Academic / research / educational use → see [LICENSE](LICENSE).
- Commercial use (product integration, defense/industrial deployment,
  commercialization of downstream academic work, FPGA/ASIC/MCU
  implementations, SaaS, etc.) → see [COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md).
- Patent notice: NanoMamba / NC-SSM and COP-RFS technologies are
  subject to pending patents (KR + US). The academic license does
  **not** grant commercial patent rights.

Commercial licensing inquiries:
**Dr. Jin Ho Choi** (SmartEAR) — `jinhochoi@smartear.co.kr`
Subject: `[Mamba-COP-RL Commercial License] <your organization>`

## Citation

If you use this code in academic work, please cite:

```bibtex
@article{choi2026mambacoprl,
  author  = {Jin Ho Choi},
  title   = {When Does Temporal Memory Help? Selective SSM vs. MLP in
             RL-Based Multi-Target DOA Track Management},
  journal = {IEEE Signal Processing Letters},
  year    = {2026},
  note    = {submitted}
}

@article{choi2015underdetermined,
  author  = {Jin Ho Choi and Chang D. Yoo},
  title   = {Underdetermined High-Resolution DOA Estimation: A
             2$\rho$th-Order Source-Signal/Noise Subspace Constrained
             Optimization},
  journal = {IEEE Transactions on Signal Processing},
  volume  = {63},
  number  = {7},
  pages   = {1858--1873},
  year    = {2015},
  doi     = {10.1109/TSP.2015.2391074}
}
```
