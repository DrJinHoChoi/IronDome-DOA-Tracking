#!/usr/bin/env python3
"""Drone "Eyes and Ears" Integrated Demo (laptop version).

Combines the full NC family on a single laptop:
  - NC-SSM / NC-TCN (KWS): "ears" -- recognises voice keywords
  - Mamba-COP-RFS (DOA + GM-PHD): "spatial awareness" -- localises and
    tracks multiple sound sources from an 8-mic array

Scenario: Search-and-Rescue (SAR)
  Drone hovers above a debris field. 3 survivors call "help" / "water" /
  "down" from different directions. One source briefly drops out
  (occluded by rubble) and reappears. A 4th source enters mid-scenario.

What the script does:
  1. Synthesise a 30-scan multi-source acoustic scene (8-mic ULA).
  2. Per scan: run COP-4th DOA + GM-PHD physics-based tracker.
  3. Per detected source: run a (stub or real) KWS classifier and
     attach a keyword label.
  4. Render a 4-panel summary figure to drone_demo.png:
        (a) raw mic-1 spectrogram
        (b) per-scan COP polar pseudo-spectrum (heatmap over time)
        (c) GM-PHD tracks colour-coded by ID with KWS captions
        (d) timeline of detected keywords per source

Usage:
    python drone_demo.py                    # synthetic SAR scenario
    python drone_demo.py --scenario combat  # 4-target jamming scenario
    python drone_demo.py --no-kws           # skip KWS (geometry only)

This is a laptop-friendly demo. No real microphone or drone needed.
For the on-device version (STM32H7 Cortex-M7 / Jetson) see cortex_m7/.

License: see LICENSE / COMMERCIAL_LICENSE.md.
Author: Jin Ho Choi (SmartEAR / NanoAgentic AI)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from iron_dome_sim.signal_model.array import UniformLinearArray
from iron_dome_sim.signal_model.signal_generator import generate_snapshots
from iron_dome_sim.doa.subspace_cop import SubspaceCOP
from iron_dome_sim.tracking.cop_phd_filter import COPPHD
from iron_dome_sim.tracking.state_models import ConstantVelocity


# --------------------------------------------------------------------- #
# Scenario definitions                                                  #
# --------------------------------------------------------------------- #
SCENARIOS = {
    "sar": {
        "name": "Search and Rescue",
        "n_scans": 30,
        # (entry_scan, exit_scan, theta_deg_t0, theta_deg_t1, keyword)
        "sources": [
            ( 0, 30, -45, -38, "help"),     # survivor 1 (slight drift)
            ( 0, 30,  +5, +12, "water"),    # survivor 2
            ( 0, 30, +50, +44, "down"),     # survivor 3
            (12, 30, -10, -20, "here"),     # survivor 4 enters at scan 12
        ],
        "occlusions": [(1, 14, 22)],   # source idx 1 invisible during 14-22
        "snr_db": 12.0,
    },
    "combat": {
        "name": "Combat Multi-Target",
        "n_scans": 30,
        "sources": [
            ( 0, 30, -30, -32, "vehicle"),
            ( 0, 30,  -5,  +5, "drone"),
            ( 0, 30, +20, +25, "shot"),
            ( 5, 25, +40, +35, "vehicle"),
        ],
        "occlusions": [(2, 10, 18)],
        "snr_db": 10.0,
    },
}

# Stub KWS labels (per-source keyword assignment from scenario).
# Replace with NC-SSM / NC-TCN inference if checkpoints are present.
KW_COLOURS = {
    "help":    "#27AE60",
    "water":   "#2980B9",
    "down":    "#E67E22",
    "here":    "#8E44AD",
    "vehicle": "#C0392B",
    "drone":   "#16A085",
    "shot":    "#D35400",
}


def build_trajectory(scenario, M=8, T=512, fov=181):
    """Synthesise per-scan multi-channel signal X (M x T) for each scan,
    plus ground-truth DOA matrix and visibility mask."""
    n_scans = scenario["n_scans"]
    sources = scenario["sources"]
    occl = {idx: (a, b) for idx, a, b in scenario["occlusions"]}

    array = UniformLinearArray(M=M)
    rng = np.random.default_rng(42)

    # Ground-truth DOAs and visibility per scan
    K_max = len(sources)
    gt_deg = np.full((n_scans, K_max), np.nan)
    gt_kw = [None] * K_max

    for i, (t_in, t_out, th0, th1, kw) in enumerate(sources):
        gt_kw[i] = kw
        for s in range(t_in, t_out):
            frac = (s - t_in) / max(t_out - t_in - 1, 1)
            th = th0 + frac * (th1 - th0)
            if i in occl:
                a, b = occl[i]
                if a <= s < b:
                    continue
            gt_deg[s, i] = th

    # Per-scan signal X
    X_per_scan = []
    for s in range(n_scans):
        active = np.where(~np.isnan(gt_deg[s]))[0]
        if len(active) == 0:
            X_per_scan.append(np.zeros((M, T), dtype=complex))
            continue
        thetas = np.deg2rad(gt_deg[s, active])
        np.random.seed(int(rng.integers(0, 1_000_000)))
        X, _, _ = generate_snapshots(array, thetas, snr_db=scenario["snr_db"],
                                     T=T, signal_type="non_stationary")
        X_per_scan.append(X)

    return array, X_per_scan, gt_deg, gt_kw


def run_pipeline(scenario, use_kws=True):
    """Run COP-DOA + GM-PHD across all scans. Returns history."""
    array, X_scans, gt_deg, gt_kw = build_trajectory(scenario)
    estimator = SubspaceCOP(array, rho=2, num_sources=None,
                            spectrum_type="combined")
    motion = ConstantVelocity(dt=1.0, process_noise_std=np.deg2rad(2.0))
    phd = COPPHD(motion_model=motion, cop_estimator=estimator,
                 use_physics=True,
                 survival_prob=0.95, detection_prob=0.9,
                 clutter_rate=1.0, birth_weight=0.12,
                 prune_threshold=1e-5)
    scan = np.linspace(-np.pi / 2, np.pi / 2, 181)

    history = {
        "spectra": [],   # per-scan COP spectrum (181,)
        "tracks":  [],   # per-scan list of (id, doa_deg)
        "gt":      gt_deg,
        "kw":      gt_kw,
        "scan_angles_deg": np.rad2deg(scan),
    }

    next_id = {}
    seen_pos_history = {}

    for s, X in enumerate(X_scans):
        _doa, P = estimator.estimate(X, scan_angles=scan)
        history["spectra"].append(P)

        ests_phd = phd.process_scan(X, scan_angles=scan)
        if isinstance(ests_phd, tuple):
            ests_phd = ests_phd[0]
        # Each entry: (state, cov, weight). state[0] = DOA in radians.
        scan_tracks = []
        for k, e in enumerate(ests_phd):
            doa = float(e[0][0])
            doa_deg = float(np.rad2deg(doa))
            # crude consistent ID by nearest existing track
            best_id = None
            best_d = 8.0
            for tid, last in seen_pos_history.items():
                if abs(last - doa_deg) < best_d:
                    best_d = abs(last - doa_deg)
                    best_id = tid
            if best_id is None:
                best_id = len(next_id)
                next_id[best_id] = best_id
            seen_pos_history[best_id] = doa_deg
            scan_tracks.append((best_id, doa_deg))
        history["tracks"].append(scan_tracks)

    return history


def attach_kws_labels(history, scenario):
    """For each track-scan, attach the nearest ground-truth keyword.
    In a real demo this is replaced by NC-SSM / NC-TCN inference on the
    beamformed audio at each track's DOA."""
    gt = history["gt"]
    kw = history["kw"]
    labelled = []
    for s, tracks in enumerate(history["tracks"]):
        per_scan = []
        for tid, deg in tracks:
            best_kw = None
            best_d = 6.0
            for k in range(gt.shape[1]):
                if np.isnan(gt[s, k]):
                    continue
                d = abs(gt[s, k] - deg)
                if d < best_d:
                    best_d = d
                    best_kw = kw[k]
            per_scan.append((tid, deg, best_kw))
        labelled.append(per_scan)
    return labelled


# --------------------------------------------------------------------- #
# Visualisation                                                         #
# --------------------------------------------------------------------- #
def render(history, labelled, scenario, out_path: Path):
    n_scans = scenario["n_scans"]
    spectra = np.stack(history["spectra"])  # (n_scans, 181)
    angles = history["scan_angles_deg"]

    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.4, 1.0],
                           hspace=0.42, wspace=0.22)

    fig.suptitle(
        f"Drone Eyes & Ears -- {scenario['name']} (8-mic ULA, "
        f"NC-SSM/NC-TCN + Mamba-COP-RFS)",
        fontsize=14, fontweight="bold")

    # ------ (a) mic-1 like proxy: time-frequency of source mixture ----
    ax = fig.add_subplot(gs[0, 0])
    proxy = 20 * np.log10(np.abs(spectra) + 1e-3)
    ax.imshow(proxy.T, aspect="auto", origin="lower",
              extent=[0, n_scans, angles[0], angles[-1]], cmap="magma")
    ax.set_title("(a) Per-scan COP pseudo-spectrum (heatmap)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Scan index")
    ax.set_ylabel("DOA (deg)")

    # ------ (b) Polar DOA at mid-scan -----------
    ax = fig.add_subplot(gs[0, 1], projection="polar")
    mid = n_scans // 2
    ax.plot(np.deg2rad(angles), spectra[mid] / spectra[mid].max(),
            color="#2C3E50", lw=1.6)
    for tid, deg, kw in labelled[mid]:
        col = KW_COLOURS.get(kw, "#888888")
        ax.plot([np.deg2rad(deg)], [1.0], 'o', color=col, ms=12,
                markeredgecolor="black", markeredgewidth=1.2)
        ax.text(np.deg2rad(deg), 1.18, kw or "?", color=col,
                fontsize=10, fontweight="bold", ha="center")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetamin(-90); ax.set_thetamax(90)
    ax.set_title(f"(b) Polar DOA + KWS @ scan {mid}",
                 fontsize=12, fontweight="bold", pad=14)

    # ------ (c) Track DOA over time, coloured by ID -----------
    ax = fig.add_subplot(gs[1, :])
    track_xy = {}
    for s, per_scan in enumerate(labelled):
        for tid, deg, kw in per_scan:
            track_xy.setdefault(tid, []).append((s, deg, kw))
    palette = plt.cm.tab10(np.linspace(0, 1, 10))
    for tid, pts in track_xy.items():
        ss = [p[0] for p in pts]; ds = [p[1] for p in pts]
        ax.plot(ss, ds, '-', color=palette[tid % 10], lw=2,
                marker='o', ms=5, label=f"track #{tid}")
        # KWS captions: only on first appearance and on changes
        prev_kw = None
        for s, d, kw in pts:
            if kw and kw != prev_kw:
                col = KW_COLOURS.get(kw, "#888888")
                ax.annotate(kw, xy=(s, d), xytext=(s, d + 8),
                            fontsize=9, color=col, fontweight="bold",
                            ha="center",
                            arrowprops=dict(arrowstyle="->",
                                             color=col, lw=0.8))
                prev_kw = kw

    # Ground-truth thin lines
    gt = history["gt"]
    for k in range(gt.shape[1]):
        ax.plot(np.arange(n_scans), gt[:, k], color="black", lw=0.7,
                linestyle="--", alpha=0.45,
                label="ground truth" if k == 0 else None)

    # Occlusion shading
    for src_idx, a, b in scenario["occlusions"]:
        ax.axvspan(a, b, alpha=0.15, color="#F39C12",
                   label="occlusion" if src_idx == scenario["occlusions"][0][0]
                                       else None)
    ax.set_title("(c) GM-PHD tracks (Mamba-COP-RFS) + KWS captions "
                 "(NC-SSM/NC-TCN)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Scan index")
    ax.set_ylabel("DOA (deg)")
    ax.set_xlim(-0.5, n_scans - 0.5)
    ax.set_ylim(angles[0], angles[-1])
    ax.legend(loc="upper right", fontsize=9, ncol=2, framealpha=0.9)
    ax.grid(alpha=0.25)

    # ------ (d) Keyword timeline per source ------------
    ax = fig.add_subplot(gs[2, :])
    for tid, pts in track_xy.items():
        for s, d, kw in pts:
            if kw is None:
                continue
            col = KW_COLOURS.get(kw, "#888888")
            ax.add_patch(Rectangle((s - 0.4, tid - 0.4), 0.8, 0.8,
                                    color=col, alpha=0.85))
            ax.text(s, tid, kw[:1].upper(), color="white",
                    fontsize=9, fontweight="bold",
                    ha="center", va="center")
    ax.set_title("(d) KWS timeline per detected track",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Scan index")
    ax.set_ylabel("Track ID")
    ax.set_xlim(-0.5, n_scans - 0.5)
    ax.set_ylim(-0.6, max(track_xy) + 0.6 if track_xy else 0.6)
    ax.grid(alpha=0.2)

    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] saved {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=list(SCENARIOS), default="sar")
    ap.add_argument("--out", type=Path, default=REPO / "drone_demo.png")
    ap.add_argument("--no-kws", action="store_true")
    args = ap.parse_args()

    scenario = SCENARIOS[args.scenario]
    print(f"=== {scenario['name']} ({scenario['n_scans']} scans, "
          f"{len(scenario['sources'])} sources) ===")
    t0 = time.time()
    history = run_pipeline(scenario, use_kws=not args.no_kws)
    labelled = attach_kws_labels(history, scenario) if not args.no_kws \
               else [[(t, d, None) for t, d in tk]
                     for tk in history["tracks"]]
    render(history, labelled, scenario, args.out)
    print(f"elapsed: {time.time()-t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
