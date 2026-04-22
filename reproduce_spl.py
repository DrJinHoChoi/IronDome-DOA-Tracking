#!/usr/bin/env python3
"""One-command reproduction of IEEE SPL 2026 paper results.

Paper:
  J. H. Choi, "When Does Temporal Memory Help? Selective SSM vs. MLP in
  RL-Based Multi-Target DOA Track Management," IEEE Signal Processing
  Letters, 2026 (submitted).

What this script does (for reviewers):
  1. Loads the pre-trained Mamba-COP-RL checkpoint (results/mamba_combat_best.pt)
  2. Evaluates it on all four CombatTrackEnv scenarios
     (Jamming, Stealth, Saturation, Formation) with fixed seeds
  3. Compares against the Fixed GM-PHD baseline (no RL)
  4. Prints a table matching Table II of the paper
  5. Optionally regenerates Fig. 2 (ablation) and Fig. 3 (tracking)
     used in the paper

Usage (reviewers -- 30 sec to ~5 min depending on flags):

    python reproduce_spl.py                        # quick (n_eval=10 seeds)
    python reproduce_spl.py --n-eval 50            # full paper setting
    python reproduce_spl.py --figs                 # also regenerate figs
    python reproduce_spl.py --full                 # also retrain MLP/GRU/LSTM
                                                   # baselines (slow, needs GPU)

Expected result (key number, Table II "Mamba (ours)" row):
    Jamming    ~0.191
    Stealth    ~0.187
    Saturation ~0.226
    Formation  ~0.248

Exit code 0 = reproduction within tolerance; 1 = large discrepancy
(see --tol flag; default 0.03 absolute GOSPA difference).

Author: Jin Ho Choi (SmartEAR)
License: see LICENSE (academic) / COMMERCIAL_LICENSE.md (commercial)
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))


# --------------------------------------------------------------------- #
# Paper's reported numbers (Table II, SPL 2026)                         #
# --------------------------------------------------------------------- #
PAPER_TABLE_II = {
    'Fixed':        {'jamming': 0.2401, 'stealth': 0.2502,
                     'saturation': 0.2571, 'formation': 0.2923},
    'Mamba (ours)': {'jamming': 0.191,  'stealth': 0.187,
                     'saturation': 0.226, 'formation': 0.248},
}


# --------------------------------------------------------------------- #
# Environment / dependency check                                        #
# --------------------------------------------------------------------- #
def check_environment() -> None:
    print("=" * 72)
    print("Mamba-COP-RL -- IEEE SPL 2026 reproduction")
    print("=" * 72)
    print(f"Python   : {platform.python_version()}")
    print(f"Platform : {platform.platform()}")

    missing = []
    for pkg in ("numpy", "scipy", "torch", "matplotlib"):
        try:
            m = __import__(pkg)
            v = getattr(m, "__version__", "?")
            print(f"{pkg:<9}: {v}")
        except ImportError:
            missing.append(pkg)
            print(f"{pkg:<9}: MISSING")

    if missing:
        print(f"\nMissing packages: {missing}")
        print("Install with: pip install -r requirements.txt")
        sys.exit(2)

    import torch
    print(f"CUDA     : {torch.cuda.is_available()} "
          f"({torch.cuda.device_count()} GPU)")
    print("=" * 72)


# --------------------------------------------------------------------- #
# Load Mamba checkpoint                                                 #
# --------------------------------------------------------------------- #
def load_mamba_policy(ckpt_path: Path):
    import torch
    from iron_dome_sim.rl.mamba_encoder import MambaCOPPolicy
    from iron_dome_sim.rl.combat_env import CombatTrackEnv

    if not ckpt_path.exists():
        print(f"\n[ERROR] Checkpoint not found: {ckpt_path}")
        print("Run training first:")
        print("  python experiments/train_combat_mamba.py")
        sys.exit(3)

    policy = MambaCOPPolicy(
        obs_dim=CombatTrackEnv.OBS_DIM,
        act_dim=CombatTrackEnv.ACT_DIM,
        d_state=24, d_hidden=48, policy_hidden=64,
    )
    state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    policy.load_state_dict(state)
    policy.eval()

    n_params = policy.param_count()
    print(f"\n[OK] Loaded {ckpt_path.name} -- "
          f"{n_params} params, {n_params/1024:.1f} KB INT8")
    return policy


# --------------------------------------------------------------------- #
# Evaluation (fixed seeds -> deterministic)                              #
# --------------------------------------------------------------------- #
def evaluate(policy, n_eval: int, n_scans: int = 200, verbose: bool = True):
    import numpy as np
    from iron_dome_sim.rl.combat_env import CombatTrackEnv, SCENARIO_TYPES

    results = {stype: {'Fixed': [], 'Mamba (ours)': []}
               for stype in SCENARIO_TYPES}

    for stype in SCENARIO_TYPES:
        if verbose:
            print(f"  [eval] {stype:<12} ({n_eval} seeds)", flush=True)
        for seed in range(n_eval):
            snr = np.random.RandomState(3000 + seed).uniform(5, 15)

            # Fixed baseline (action = 0 -> default GM-PHD params)
            env = CombatTrackEnv(M=8, snr_db=snr, T=64,
                                 n_scans=n_scans, scenario_type=stype)
            env.reset(seed=3000 + seed)
            g = []
            for _ in range(n_scans):
                _, _, done, info = env.step(np.zeros(CombatTrackEnv.ACT_DIM))
                g.append(info['gospa'])
                if done:
                    break
            results[stype]['Fixed'].append(float(np.mean(g)))

            # Mamba-COP-RL
            env = CombatTrackEnv(M=8, snr_db=snr, T=64,
                                 n_scans=n_scans, scenario_type=stype)
            obs = env.reset(seed=3000 + seed)
            if hasattr(policy, 'reset_hidden'):
                policy.reset_hidden()
            g = []
            for _ in range(n_scans):
                a, _, _ = policy.get_action(obs, deterministic=True)
                obs, _, done, info = env.step(a)
                g.append(info['gospa'])
                if done:
                    break
            results[stype]['Mamba (ours)'].append(float(np.mean(g)))

    return results


# --------------------------------------------------------------------- #
# Report                                                                #
# --------------------------------------------------------------------- #
def report(results: dict, tol: float) -> int:
    import numpy as np
    print("\n" + "=" * 72)
    print("Table II (reproduction) -- Per-Scenario Mean GOSPA")
    print("=" * 72)
    print(f"{'Scenario':<12} {'Fixed':>10} {'Fixed paper':>12} "
          f"{'Mamba':>10} {'Mamba paper':>12} {'|d| vs paper':>12}")
    print("-" * 72)

    max_dev = 0.0
    for stype in ('jamming', 'stealth', 'saturation', 'formation'):
        fixed  = float(np.mean(results[stype]['Fixed']))
        mamba  = float(np.mean(results[stype]['Mamba (ours)']))
        p_fix  = PAPER_TABLE_II['Fixed'][stype]
        p_mam  = PAPER_TABLE_II['Mamba (ours)'][stype]
        dev    = abs(mamba - p_mam)
        max_dev = max(max_dev, dev)
        flag   = "" if dev <= tol else " !"
        print(f"{stype:<12} {fixed:>10.4f} {p_fix:>12.4f} "
              f"{mamba:>10.4f} {p_mam:>12.4f} {dev:>+12.4f}{flag}")
    print("=" * 72)

    ok = max_dev <= tol
    print(f"\nMax ||d| vs paper| = {max_dev:.4f}  (tolerance = {tol})")
    if ok:
        print("[PASS] REPRODUCTION PASSED -- numbers match the paper within tolerance.")
        return 0
    else:
        print("[FAIL] REPRODUCTION FAILED -- at least one scenario exceeds tolerance.")
        print("   (Note: some variance is expected due to non-determinism in")
        print("    the signal-generation seed chain; try --n-eval 50.)")
        return 1


# --------------------------------------------------------------------- #
# Optional: regenerate paper figures                                    #
# --------------------------------------------------------------------- #
def regenerate_figs() -> None:
    print("\n[figs] Regenerating fig_architecture + fig_tracking ...")
    import subprocess
    spl_dir = REPO_ROOT / "paper" / "spl2026"
    for script in ("make_fig_architecture.py", "make_fig_tracking.py"):
        p = spl_dir / script
        if p.exists():
            subprocess.run([sys.executable, str(p)], cwd=spl_dir, check=False)
        else:
            print(f"  [skip] {p} not found")


# --------------------------------------------------------------------- #
# Entry                                                                 #
# --------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(
        description="IEEE SPL 2026 reproduction -- one-command reviewer script"
    )
    ap.add_argument("--ckpt", type=Path,
                    default=REPO_ROOT / "results" / "mamba_combat_best.pt",
                    help="Path to Mamba-COP-RL checkpoint (.pt).")
    ap.add_argument("--n-eval", type=int, default=10,
                    help="Seeds per scenario (paper uses 50).")
    ap.add_argument("--n-scans", type=int, default=200,
                    help="Scans per episode (paper: 200).")
    ap.add_argument("--tol", type=float, default=0.03,
                    help="Absolute GOSPA tolerance vs paper Table II.")
    ap.add_argument("--figs", action="store_true",
                    help="Also regenerate fig_architecture and fig_tracking.")
    ap.add_argument("--full", action="store_true",
                    help="Also retrain MLP/GRU/LSTM baselines (slow).")
    args = ap.parse_args()

    check_environment()
    t0 = time.time()

    policy = load_mamba_policy(args.ckpt)

    print(f"\nEvaluating {args.n_eval} seeds x 4 scenarios x {args.n_scans} scans ...")
    results = evaluate(policy, n_eval=args.n_eval, n_scans=args.n_scans)

    code = report(results, tol=args.tol)
    print(f"\nTotal elapsed: {time.time()-t0:.1f} s")

    if args.figs:
        regenerate_figs()

    if args.full:
        print("\n[full] Retraining MLP / GRU / LSTM baselines ...")
        import subprocess
        subprocess.run(
            [sys.executable, str(REPO_ROOT / "experiments" / "train_rl_tracker.py")],
            check=False,
        )

    return code


if __name__ == "__main__":
    sys.exit(main())
