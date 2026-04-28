#!/usr/bin/env python3
"""Drone Eyes & Ears -- interactive Streamlit dashboard.

Live, slider-driven companion to drone_demo.py for pitching/demos.
Reuses build_trajectory / run_pipeline / attach_kws_labels from
drone_demo.py so there is a single source of truth for the simulation.

Launch:
    streamlit run drone_demo_streamlit.py

Pipeline summary:
    8-mic ULA -> COP-4th DOA -> GM-PHD (Mamba-COP-RFS) -> NC-SSM/NC-TCN KWS.
    Footprint: 41.4 KB INT8 / 3.4 ms per scan.

License: dual academic + commercial. See LICENSE / COMMERCIAL_LICENSE.md.
Author: Jin Ho Choi (SmartEAR / NanoAgentic AI).
"""

# requirements: streamlit>=1.30

from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import streamlit as st

from drone_demo import (
    SCENARIOS,
    KW_COLOURS,
    build_trajectory,  # noqa: F401  (kept for downstream extension)
    run_pipeline,
    attach_kws_labels,
)


# --------------------------------------------------------------------- #
# Cached pipeline wrapper                                               #
# --------------------------------------------------------------------- #
def _scenario_with_overrides(scenario_key: str, snr_db: float,
                             n_scans: int) -> dict:
    """Return a deep-copied scenario dict with SNR / n_scans overrides.

    Sources whose entry/exit times exceed n_scans are clipped so the
    simulation stays internally consistent.
    """
    base = copy.deepcopy(SCENARIOS[scenario_key])
    base["snr_db"] = float(snr_db)
    base["n_scans"] = int(n_scans)

    clipped_sources = []
    for (t_in, t_out, th0, th1, kw) in base["sources"]:
        t_in_c = max(0, min(t_in, n_scans))
        t_out_c = max(t_in_c + 1, min(t_out, n_scans))
        if t_in_c >= n_scans:
            continue
        clipped_sources.append((t_in_c, t_out_c, th0, th1, kw))
    base["sources"] = clipped_sources

    clipped_occl = []
    for (idx, a, b) in base["occlusions"]:
        if idx >= len(clipped_sources):
            continue
        a_c = max(0, min(a, n_scans))
        b_c = max(a_c, min(b, n_scans))
        clipped_occl.append((idx, a_c, b_c))
    base["occlusions"] = clipped_occl

    return base


@st.cache_data(show_spinner="Running COP-DOA + GM-PHD pipeline...")
def cached_run(scenario_key: str, snr_db: float, n_scans: int,
               nonce: int = 0):
    """Cached wrapper around run_pipeline + attach_kws_labels.

    `nonce` is bumped by the "Re-run simulation" button so users can
    force a fresh draw of the random seed without changing parameters.
    """
    scenario = _scenario_with_overrides(scenario_key, snr_db, n_scans)
    t0 = time.time()
    history = run_pipeline(scenario)
    labelled = attach_kws_labels(history, scenario)
    elapsed = time.time() - t0
    return scenario, history, labelled, elapsed


# --------------------------------------------------------------------- #
# Plot helpers (matplotlib, matched to drone_demo.render style)         #
# --------------------------------------------------------------------- #
def fig_polar(history, labelled, scan_idx: int):
    angles = history["scan_angles_deg"]
    spectra = history["spectra"]
    P = spectra[scan_idx]
    P_norm = P / (P.max() + 1e-9)

    fig = plt.figure(figsize=(5.2, 5.2))
    ax = fig.add_subplot(111, projection="polar")
    ax.plot(np.deg2rad(angles), P_norm, color="#2C3E50", lw=1.6)
    for tid, deg, kw in labelled[scan_idx]:
        col = KW_COLOURS.get(kw, "#888888")
        ax.plot([np.deg2rad(deg)], [1.0], "o", color=col, ms=12,
                markeredgecolor="black", markeredgewidth=1.2)
        ax.text(np.deg2rad(deg), 1.18, kw or "?", color=col,
                fontsize=10, fontweight="bold", ha="center")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetamin(-90)
    ax.set_thetamax(90)
    ax.set_title(f"Polar DOA + KWS @ scan {scan_idx}",
                 fontsize=11, fontweight="bold", pad=12)
    fig.tight_layout()
    return fig


def fig_spectrum(history, scan_idx: int):
    angles = history["scan_angles_deg"]
    P = history["spectra"][scan_idx]
    P_db = 20.0 * np.log10(np.abs(P) + 1e-6)

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot(angles, P_db, color="#1F77B4", lw=1.6)
    ax.fill_between(angles, P_db.min(), P_db, color="#1F77B4", alpha=0.18)
    ax.set_xlabel("DOA (deg)")
    ax.set_ylabel("COP power (dB)")
    ax.set_title(f"COP pseudo-spectrum @ scan {scan_idx}",
                 fontsize=11, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.set_xlim(angles[0], angles[-1])
    fig.tight_layout()
    return fig


def fig_tracks(history, labelled, scenario, scan_idx: int,
               show_gt: bool, show_occl: bool):
    n_scans = scenario["n_scans"]
    angles = history["scan_angles_deg"]

    track_xy = {}
    for s, per_scan in enumerate(labelled):
        for tid, deg, kw in per_scan:
            track_xy.setdefault(tid, []).append((s, deg, kw))

    fig, ax = plt.subplots(figsize=(11, 4.2))
    palette = plt.cm.tab10(np.linspace(0, 1, 10))
    for tid, pts in track_xy.items():
        ss = [p[0] for p in pts]
        ds = [p[1] for p in pts]
        ax.plot(ss, ds, "-", color=palette[tid % 10], lw=2,
                marker="o", ms=5, label=f"track #{tid}")
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

    if show_gt:
        gt = history["gt"]
        for k in range(gt.shape[1]):
            ax.plot(np.arange(n_scans), gt[:, k], color="black", lw=0.7,
                    linestyle="--", alpha=0.45,
                    label="ground truth" if k == 0 else None)

    if show_occl:
        first = scenario["occlusions"][0][0] if scenario["occlusions"] else None
        for src_idx, a, b in scenario["occlusions"]:
            ax.axvspan(a, b, alpha=0.15, color="#F39C12",
                       label="occlusion" if src_idx == first else None)

    ax.axvline(scan_idx, color="#C0392B", lw=1.6, alpha=0.85,
               label=f"current scan = {scan_idx}")
    ax.set_title("GM-PHD tracks (Mamba-COP-RFS) + KWS captions",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Scan index")
    ax.set_ylabel("DOA (deg)")
    ax.set_xlim(-0.5, n_scans - 0.5)
    ax.set_ylim(angles[0], angles[-1])
    ax.legend(loc="upper right", fontsize=9, ncol=2, framealpha=0.9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def fig_kws_timeline(labelled, scenario):
    n_scans = scenario["n_scans"]

    track_xy = {}
    for s, per_scan in enumerate(labelled):
        for tid, deg, kw in per_scan:
            track_xy.setdefault(tid, []).append((s, deg, kw))

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
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
    ax.set_title("KWS timeline per detected track",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Scan index")
    ax.set_ylabel("Track ID")
    ax.set_xlim(-0.5, n_scans - 0.5)
    ax.set_ylim(-0.6, max(track_xy) + 0.6 if track_xy else 0.6)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------- #
# Stats                                                                 #
# --------------------------------------------------------------------- #
def compute_gospa(history, c: float = 10.0) -> float:
    """Cheap GOSPA proxy: mean per-scan |closest gt - estimate| with
    cardinality penalty c for unmatched. Sufficient for dashboard stat."""
    gt = history["gt"]
    tracks = history["tracks"]
    if gt.size == 0 or len(tracks) == 0:
        return float("nan")
    vals = []
    for s, scan_tracks in enumerate(tracks):
        gt_s = gt[s]
        gt_active = gt_s[~np.isnan(gt_s)]
        ests = np.array([d for _, d in scan_tracks], dtype=float)
        if gt_active.size == 0 and ests.size == 0:
            vals.append(0.0)
            continue
        if ests.size == 0 or gt_active.size == 0:
            vals.append(c)
            continue
        diffs = np.abs(gt_active[:, None] - ests[None, :])
        closest = diffs.min(axis=1)
        unmatched = max(ests.size - gt_active.size, 0)
        vals.append(closest.mean() + c * unmatched / max(gt_active.size, 1))
    return float(np.mean(vals))


# --------------------------------------------------------------------- #
# Page                                                                  #
# --------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(page_title="Drone Eyes & Ears", layout="wide")

    # ---- Sidebar ----
    st.sidebar.title("Controls")
    scenario_key = st.sidebar.radio(
        "Scenario",
        options=["sar", "combat", "custom"],
        index=0,
        format_func=lambda k: {
            "sar": "Search and Rescue",
            "combat": "Combat Multi-Target",
            "custom": "Custom (sar base, your SNR/scans)",
        }[k],
    )
    effective_key = "sar" if scenario_key == "custom" else scenario_key

    snr_db = st.sidebar.slider("SNR (dB)", min_value=0.0, max_value=20.0,
                               value=float(SCENARIOS[effective_key]["snr_db"]),
                               step=0.5)
    n_scans = st.sidebar.slider("Number of scans", min_value=10, max_value=60,
                                value=int(SCENARIOS[effective_key]["n_scans"]),
                                step=1)
    show_gt = st.sidebar.toggle("Show ground truth", value=True)
    show_occl = st.sidebar.toggle("Show occlusion shading", value=True)

    if "rerun_nonce" not in st.session_state:
        st.session_state.rerun_nonce = 0
    if st.sidebar.button("Re-run simulation", type="primary"):
        st.session_state.rerun_nonce += 1

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Cached results: only re-runs on parameter or button change. "
        "Slider movement is instant."
    )

    # ---- Run pipeline (cached) ----
    scenario, history, labelled, elapsed = cached_run(
        effective_key, snr_db, n_scans, st.session_state.rerun_nonce
    )

    # ---- Header ----
    st.title("Drone Eyes & Ears -- Live Dashboard")
    st.markdown(
        f"**Scenario:** {scenario['name']}  |  "
        f"**Array:** 8-mic ULA  |  "
        f"**Pipeline:** COP-4th DOA -> GM-PHD (Mamba-COP-RFS) -> "
        f"NC-SSM / NC-TCN KWS"
    )

    # Keyword colour chips (only those active in this scenario)
    active_kws = {kw for (_, _, _, _, kw) in scenario["sources"]}
    chip_html = []
    for kw in sorted(active_kws):
        col = KW_COLOURS.get(kw, "#888888")
        chip_html.append(
            f"<span style='background:{col};color:white;padding:3px 10px;"
            f"border-radius:12px;margin-right:6px;font-weight:600;"
            f"font-size:0.85em'>{kw}</span>"
        )
    if chip_html:
        st.markdown("".join(chip_html), unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Footprint", "41.4 KB", "INT8")
    c2.metric("Per-scan latency", "3.4 ms", "Cortex-M7")
    c3.metric("Scans simulated", f"{n_scans}")
    c4.metric("Pipeline elapsed", f"{elapsed:.2f} s")

    st.markdown("---")

    # ---- Big slider ----
    scan_idx = st.slider("Scan index (drives all panels)",
                         min_value=0, max_value=max(n_scans - 1, 0),
                         value=min(n_scans // 2, max(n_scans - 1, 0)),
                         step=1)

    # ---- Row 1: polar + spectrum ----
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.pyplot(fig_polar(history, labelled, scan_idx),
                  clear_figure=True)
    with r1c2:
        st.pyplot(fig_spectrum(history, scan_idx), clear_figure=True)

    # ---- Row 2: full-width tracks ----
    st.pyplot(
        fig_tracks(history, labelled, scenario, scan_idx,
                   show_gt=show_gt, show_occl=show_occl),
        clear_figure=True,
    )

    # ---- Row 3: KWS timeline + stats ----
    r3c1, r3c2 = st.columns([1.2, 1.0])
    with r3c1:
        st.pyplot(fig_kws_timeline(labelled, scenario), clear_figure=True)
    with r3c2:
        st.subheader("Stats")
        cur_tracks = labelled[scan_idx]
        n_det = len(cur_tracks)
        gospa = compute_gospa(history)
        st.metric("Detected tracks @ current scan", f"{n_det}")
        st.metric("Mean GOSPA (deg)",
                  f"{gospa:.2f}" if not np.isnan(gospa) else "n/a")
        st.metric("Latency budget / scan", "3.4 ms",
                  delta="-96.6 ms vs 100 ms target",
                  delta_color="normal")

        if cur_tracks:
            st.markdown("**Tracks at this scan:**")
            for tid, deg, kw in cur_tracks:
                col = KW_COLOURS.get(kw, "#888888")
                st.markdown(
                    f"- track #{tid}: DOA = `{deg:+6.1f} deg`  "
                    f"<span style='background:{col};color:white;"
                    f"padding:1px 8px;border-radius:8px;font-size:0.85em'>"
                    f"{kw or '?'}</span>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("No active tracks at this scan.")

    # ---- Footer ----
    st.markdown("---")
    st.caption(
        "Code: github.com/DrJinHoChoi/IronDome-DOA-Tracking | "
        "License: dual academic+commercial | "
        "Contact: jinhochoi@smartear.co.kr"
    )


if __name__ == "__main__":
    main()
