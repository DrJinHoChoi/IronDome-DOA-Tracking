# IEEE SPL 2026 — Reviewer Guide

> **Paper**: *"When Does Temporal Memory Help? Selective SSM vs. MLP in
> RL-Based Multi-Target DOA Track Management"*
> **Author**: Jin Ho Choi (SmartEAR / NanoAgentic AI)
> **Venue**: IEEE Signal Processing Letters, 2026 (under review)

This page is the **single entry point for SPL reviewers**. Everything
not on this list is supplementary (other companion papers also live in
this repository — see *Repository scope* at the bottom).

---

## :rocket: Quick reproduction (one command, ~30 s)

```bash
git clone https://github.com/DrJinHoChoi/IronDome-DOA-Tracking
cd IronDome-DOA-Tracking
pip install -r requirements.txt
python reproduce_spl.py                # quick (10 seeds/scenario)
python reproduce_spl.py --n-eval 50    # full paper setting (Table II)
```

The script:
1. Verifies environment (Python, torch, numpy, scipy, matplotlib).
2. Loads the pre-trained Mamba-COP-RL checkpoint
   (`results/mamba_combat_best.pt`, 42K params, 41.4 KB INT8).
3. Evaluates on all 4 CombatTrackEnv scenarios with deterministic seeds.
4. Compares against the Fixed GM-PHD baseline.
5. **Prints a table matching Table II** of the manuscript.
6. **Exit code 0** if measured GOSPA falls within `--tol` (default 0.03)
   of paper values; **exit 1** otherwise.

Expected key numbers (Table II, Mamba row):

| Scenario | Paper | Tolerance |
|---|---|---|
| Jamming | 0.191 | 0.03 |
| Stealth | 0.187 | 0.03 |
| Saturation | 0.226 | 0.03 |
| Formation | 0.248 | 0.03 |

---

## :file_folder: Files relevant to the SPL submission

### Manuscript & figures
| Path | Description |
|---|---|
| [`paper/spl2026/mamba_cop_rl_spl.tex`](paper/spl2026/mamba_cop_rl_spl.tex) | LaTeX source (5 p body + refs) |
| [`paper/spl2026/mamba_cop_rl_spl.bib`](paper/spl2026/mamba_cop_rl_spl.bib) | Bibliography (16 refs, all verified) |
| [`paper/spl2026/mamba_cop_rl_spl.pdf`](paper/spl2026/mamba_cop_rl_spl.pdf) | Compiled PDF |
| [`paper/spl2026/fig_architecture.pdf`](paper/spl2026/fig_architecture.pdf) | Fig. 1 — pipeline architecture |
| [`paper/spl2026/fig_ablation.pdf`](paper/spl2026/fig_ablation.pdf) | Fig. 2 — Proposition 1 ablation |
| [`paper/spl2026/fig_tracking.pdf`](paper/spl2026/fig_tracking.pdf) | Fig. 3 — stealth scenario tracking |
| [`paper/spl2026/make_fig_architecture.py`](paper/spl2026/make_fig_architecture.py) | Fig. 1 generator |
| [`paper/spl2026/make_fig_tracking.py`](paper/spl2026/make_fig_tracking.py) | Fig. 3 generator |

### Code (everything needed to reproduce)
| Path | What it implements |
|---|---|
| [`reproduce_spl.py`](reproduce_spl.py) | One-command reviewer entry point |
| [`iron_dome_sim/rl/mamba_encoder.py`](iron_dome_sim/rl/mamba_encoder.py) | Selective SSM encoder (Eq. 7) + actor-critic policy |
| [`iron_dome_sim/rl/combat_env.py`](iron_dome_sim/rl/combat_env.py) | CombatTrackEnv: 4 scenarios with temporal-dependency horizons |
| [`iron_dome_sim/rl/policy.py`](iron_dome_sim/rl/policy.py) | PPO trainer + rollout buffer |
| [`iron_dome_sim/doa/subspace_cop.py`](iron_dome_sim/doa/subspace_cop.py) | COP-4th DOA estimator (181-dim spectrum) |
| [`iron_dome_sim/tracking/cop_phd_filter.py`](iron_dome_sim/tracking/cop_phd_filter.py) | GM-PHD with predict–identify–update pipeline |
| [`experiments/train_combat_mamba.py`](experiments/train_combat_mamba.py) | Full training script (300 episodes × 200 scans) |
| [`experiments/train_rl_tracker.py`](experiments/train_rl_tracker.py) | Baseline trainers (MLP-PPO, GRU-PPO, LSTM-PPO) |
| [`experiments/temporal_cop_vs_baseline.py`](experiments/temporal_cop_vs_baseline.py) | Proposition 1 ablation (12-dim vs 183-dim observation) |

### Pre-trained checkpoint
| Path | Provenance |
|---|---|
| `results/mamba_combat_best.pt` | Best of 300 PPO episodes (entropy 0.02, lr 3e-4); 42,375 params, 41.4 KB INT8 |

### Submission artefacts
| Path |
|---|
| [`paper/spl2026/mamba_cop_rl_spl_submit.zip`](paper/spl2026/mamba_cop_rl_spl_submit.zip) — exactly what's uploaded to ScholarOne |

---

## :clipboard: Mapping paper claims → code

| Paper claim | Code path |
|---|---|
| Proposition 1 (sufficient-statistic condition) | Proven in §IV-A; ablation in [`experiments/temporal_cop_vs_baseline.py`](experiments/temporal_cop_vs_baseline.py) |
| Algorithm 1 (rollout + PPO update) | [`iron_dome_sim/rl/policy.py`](iron_dome_sim/rl/policy.py) lines 1–240; rollout in [`experiments/train_combat_mamba.py`](experiments/train_combat_mamba.py) |
| Eq. 7 (SSM encoder) | [`iron_dome_sim/rl/mamba_encoder.py`](iron_dome_sim/rl/mamba_encoder.py) `MambaSSMEncoder.forward()` |
| Table II (per-scenario GOSPA) | Reproduced exactly by `reproduce_spl.py --n-eval 50` |
| Table III (overall GOSPA + d_s ablation) | Run `python experiments/train_combat_mamba.py --d-state 8/16/24/32` |
| Table IV (obs dim × episode length ablation) | Run `python experiments/temporal_cop_vs_baseline.py` |
| Table V (latency on STM32H7) | [`cortex_m7/`](cortex_m7/) — INT8-quantised C kernels with cycle counts |
| Fig. 3 (stealth tracking visualisation) | Generated by [`paper/spl2026/make_fig_tracking.py`](paper/spl2026/make_fig_tracking.py) |

---

## :wrench: Environment

- **OS**: tested on Windows 11 + Ubuntu 22.04
- **Python**: 3.10 / 3.11 / 3.13
- **Key deps**: `torch>=2.0`, `numpy`, `scipy`, `matplotlib`
- **CPU-only is fine** for `reproduce_spl.py` (no GPU required for evaluation)
- **GPU recommended only** for `--full` retraining of baselines

---

## :pushpin: Submission commit

The exact commit that matches the submitted PDF is tagged:

```bash
git checkout spl2026-v1
```

Tag: `spl2026-v1` (created at the time of submission to ScholarOne).

---

## :package: Repository scope (other content reviewers can ignore)

This repository hosts **two related papers** that share the COP-RFS
simulator, the GM-PHD tracker implementation, and pre-trained
checkpoints (Option A: shared repo). For SPL reviewing, **only the
files listed above are relevant**.

The other paper present in the repo:

- `paper/cop_rfs_tsp2026.tex` — companion **IEEE TSP** journal
  manuscript ("Underdetermined High-Resolution DOA Estimation and
  Multi-Target Tracking via COP-RFS") — focuses on the underlying
  signal-processing pipeline (T-COP, SD-COP, predict-identify-update
  GM-PHD). The SPL letter (this submission) layers an RL-based
  GM-PHD-parameter scheduler on top of that pipeline.

Auxiliary / off-topic for SPL:

- `drone_demo*.py`, `drone_demo*.ipynb` — laptop demo (eyes & ears)
- `cortex_m7/` — embedded INT8 C kernels (latency measurements only)
- `tests/` — unit tests for COP, MUSIC, ESPRIT
- `paper/icassp2027/` — ICASSP 2027 extended abstract (separate venue)

---

## :scroll: License

Dual-licensed:
- **Academic / non-commercial use**: free under [LICENSE](LICENSE).
- **Commercial use**: separate license — see [COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md).
- **Patent notice**: NanoMamba / NC-SSM (KR + US pending) and
  COP-RFS (KR pending). Academic license does **not** grant
  commercial patent rights.

---

## :email: Contact for reviewers

If reproduction fails or any path above is unclear, please raise it
in the SPL ScholarOne review system, or contact:

> **Dr. Jin Ho Choi** — SmartEAR / NanoAgentic AI
> jinhochoi@smartear.co.kr
