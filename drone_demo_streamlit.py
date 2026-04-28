"""Drone Fleet Operations Console -- Browser Dashboard.

Multi-drone tactical view powered by the NC family:
    eyes  : NC-Conv-SSM (vision)
    ears  : NC-SSM / NC-TCN (KWS)
    spatial: Mamba-COP-RFS (8-mic ULA DOA + GM-PHD tracking)

Layout:
    +--------------------------------------------------------+
    |  FLEET STATUS BAR -- N drones, click to select         |
    +--------------------------------------------------------+
    |  TACTICAL MAP (bird's-eye)         |  SELECTED DRONE   |
    |  - drone icons + heading           |  - eye (live cam) |
    |  - detection cones                 |  - ear (live mic) |
    |  - tracked targets                 |  - status         |
    +--------------------------------------------------------+
    |  POLAR 360 of selected drone       |  TIMELINE         |
    +--------------------------------------------------------+
    |  FLEET EVENT LOG                                       |
    +--------------------------------------------------------+

Run:
    streamlit run drone_demo_streamlit.py
"""
from __future__ import annotations

import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from drone_demo import (SCENARIOS, KW_COLOURS, run_pipeline,
                        attach_kws_labels)


st.set_page_config(page_title="Drone Fleet Console",
                   page_icon=":helicopter:", layout="wide")


# --------------------------------------------------------------------- #
# Fleet definition                                                      #
# --------------------------------------------------------------------- #
def default_fleet():
    return [
        {"id": "Alpha-1", "pos": (-30, 20), "heading": 30,
         "scenario": "sar", "live_cam": True,  "live_mic": True,
         "color": "#27AE60"},
        {"id": "Alpha-2", "pos": (15, 35),   "heading": 270,
         "scenario": "sar", "live_cam": False, "live_mic": False,
         "color": "#2980B9"},
        {"id": "Bravo-1", "pos": (40, -10),  "heading": 200,
         "scenario": "combat", "live_cam": False, "live_mic": False,
         "color": "#E67E22"},
        {"id": "Bravo-2", "pos": (-15, -25), "heading": 110,
         "scenario": "combat", "live_cam": False, "live_mic": False,
         "color": "#C0392B"},
    ]


def _scenario_with_overrides(key: str, snr_db: float, n_scans: int) -> dict:
    base = deepcopy(SCENARIOS[key])
    base["snr_db"] = float(snr_db)
    base["n_scans"] = int(n_scans)
    base["sources"] = [(min(t_in, n_scans - 1), min(t_out, n_scans),
                        th0, th1, kw)
                       for t_in, t_out, th0, th1, kw in base["sources"]]
    base["occlusions"] = [(idx, min(a, n_scans), min(b, n_scans))
                          for idx, a, b in base["occlusions"]]
    return base


@st.cache_data(show_spinner="Running spatial pipeline...")
def cached_run(scenario_key, snr_db, n_scans, kws_mode, kws_model,
               kws_backbone, drone_id, nonce=0):
    sc = _scenario_with_overrides(scenario_key, snr_db, n_scans)
    history = run_pipeline(sc)
    labelled = attach_kws_labels(history, sc,
                                 real_kws=(kws_mode == "real"),
                                 model=kws_model, backbone=kws_backbone)
    return sc, history, labelled


# --------------------------------------------------------------------- #
# Plot helpers                                                          #
# --------------------------------------------------------------------- #
def fig_tactical_map(fleet, selected_id, fleet_data, scan_idx):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor("#1A1F2E")
    fig.patch.set_facecolor("#1A1F2E")
    ax.set_xlim(-60, 60); ax.set_ylim(-50, 50)
    ax.set_aspect("equal")

    # Grid
    for r in (10, 20, 30, 40):
        circ = plt.Circle((0, 0), r, fill=False, color="#2C3E50",
                          lw=0.8, alpha=0.6, linestyle="--")
        ax.add_patch(circ)
    ax.axhline(0, color="#2C3E50", lw=0.5, alpha=0.4)
    ax.axvline(0, color="#2C3E50", lw=0.5, alpha=0.4)

    # Targets aggregated from all fleet detections at scan_idx
    target_x, target_y, target_kw = [], [], []
    for d in fleet:
        sc, hist, lab = fleet_data[d["id"]]
        if scan_idx >= len(lab):
            continue
        dx, dy = d["pos"]
        # Each detection becomes a relative bearing from drone position
        for tid, deg, kw in lab[scan_idx]:
            r = 25  # nominal range
            theta_world = np.deg2rad(d["heading"] + deg)
            tx = dx + r * np.cos(theta_world)
            ty = dy + r * np.sin(theta_world)
            target_x.append(tx); target_y.append(ty)
            target_kw.append(kw)
            # Detection ray
            ax.plot([dx, tx], [dy, ty],
                    color=KW_COLOURS.get(kw, "#888"),
                    lw=1.0, alpha=0.45, linestyle=":")

    # Draw targets
    for x, y, kw in zip(target_x, target_y, target_kw):
        col = KW_COLOURS.get(kw, "#888")
        ax.scatter([x], [y], s=120, color=col, marker="X",
                   edgecolors="white", linewidths=1.5, zorder=5)

    # Draw drones
    for d in fleet:
        x, y = d["pos"]
        is_sel = (d["id"] == selected_id)
        # Detection cone (FOV ~180 deg in the heading direction)
        theta = np.deg2rad(d["heading"])
        for sweep in np.linspace(-np.pi/2, np.pi/2, 15):
            ax.plot([x, x + 30*np.cos(theta + sweep)],
                    [y, y + 30*np.sin(theta + sweep)],
                    color=d["color"], lw=0.4, alpha=0.10)
        # Triangle drone marker
        size = 4
        tri = np.array([[size, 0], [-size*0.6, size*0.6],
                         [-size*0.6, -size*0.6]])
        c, s = np.cos(theta), np.sin(theta)
        rot = np.array([[c, -s], [s, c]])
        tri = (rot @ tri.T).T + np.array([x, y])
        poly = plt.Polygon(tri, color=d["color"],
                            ec="white" if is_sel else "none",
                            lw=2.5 if is_sel else 0, zorder=6)
        ax.add_patch(poly)
        # Label
        ax.text(x, y - 6, d["id"], color=d["color"],
                fontsize=9, fontweight="bold", ha="center",
                bbox=dict(facecolor="#0E1117", edgecolor="none",
                          alpha=0.75, pad=1.5))
        if is_sel:
            ax.scatter([x], [y], s=400, facecolors="none",
                       edgecolors="#F1C40F", lw=2, zorder=7)

    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"Tactical Map  --  scan {scan_idx}  --  "
                 f"{len(fleet)} drones, {len(target_x)} contacts",
                 color="white", fontsize=12, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_color("#2C3E50")
    fig.tight_layout()
    return fig


def fig_polar_for(history, labelled, scan_idx, drone_color):
    angles = history["scan_angles_deg"]
    P = history["spectra"][scan_idx]
    P = P / (P.max() + 1e-9)
    fig = plt.figure(figsize=(5.8, 4.6))
    fig.patch.set_facecolor("#0E1117")
    ax = fig.add_subplot(111, projection="polar")
    ax.set_facecolor("#1A1F2E")
    ax.plot(np.deg2rad(angles), P, color=drone_color, lw=1.6,
            alpha=0.85)
    ax.fill(np.deg2rad(angles), P, color=drone_color, alpha=0.12)
    for tid, deg, kw in labelled[scan_idx]:
        col = KW_COLOURS.get(kw, "#888")
        ax.plot([np.deg2rad(deg)], [1.0], "o", color=col, ms=12,
                markeredgecolor="white", markeredgewidth=1.5)
        ax.text(np.deg2rad(deg), 1.22, kw or "?", color=col,
                fontweight="bold", fontsize=10, ha="center")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetamin(-90); ax.set_thetamax(90)
    ax.set_yticklabels([])
    ax.tick_params(colors="white")
    ax.set_title(f"DOA spectrum (8-mic ULA)",
                 color="white", fontsize=11, fontweight="bold", pad=10)
    fig.tight_layout()
    return fig


def fig_timeline(history, labelled, scenario, scan_idx, color,
                 show_gt, show_occl):
    n = scenario["n_scans"]
    angles = history["scan_angles_deg"]
    fig, ax = plt.subplots(figsize=(13, 2.6))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#1A1F2E")
    palette = plt.cm.tab10(np.linspace(0, 1, 10))
    by_id = {}
    for s, per in enumerate(labelled):
        for tid, deg, kw in per:
            by_id.setdefault(tid, []).append((s, deg, kw))
    for tid, pts in by_id.items():
        ss = [p[0] for p in pts]; ds = [p[1] for p in pts]
        ax.plot(ss, ds, "-", color=palette[tid % 10], lw=1.8,
                marker="o", ms=3)
        prev = None
        for s, d, kw in pts:
            if kw and kw != prev:
                col = KW_COLOURS.get(kw, "#888")
                ax.text(s, d + 6, kw, fontsize=8, color=col,
                        fontweight="bold", ha="center")
                prev = kw
    if show_gt:
        gt = history["gt"]
        for k in range(gt.shape[1]):
            ax.plot(np.arange(n), gt[:, k], color="#7F8C8D",
                    lw=0.6, linestyle="--", alpha=0.4)
    if show_occl:
        for sidx, a, b in scenario["occlusions"]:
            ax.axvspan(a, b, alpha=0.18, color="#F39C12")
    ax.axvline(scan_idx, color=color, lw=2, alpha=0.8)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(angles[0], angles[-1])
    ax.set_xlabel("Scan", color="white")
    ax.set_ylabel("DOA (deg)", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#2C3E50")
    ax.grid(alpha=0.2, color="#2C3E50")
    ax.set_title("Track + KWS timeline (selected drone)",
                 color="white", fontsize=10, fontweight="bold")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------- #
# Live capture                                                          #
# --------------------------------------------------------------------- #
def capture_camera(idx: int):
    import cv2
    cam = cv2.VideoCapture(int(idx))
    ok, frame = cam.read()
    cam.release()
    if not ok or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), frame


def capture_mic(seconds: float, channels: int, device: int | None):
    import sounddevice as sd
    n = int(seconds * 16000)
    return sd.rec(n, samplerate=16000, channels=channels,
                  device=device, dtype="float32", blocking=True)


# --------------------------------------------------------------------- #
# Page                                                                  #
# --------------------------------------------------------------------- #
def main():
    if "fleet" not in st.session_state:
        st.session_state.fleet = default_fleet()
    if "selected" not in st.session_state:
        st.session_state.selected = st.session_state.fleet[0]["id"]
    if "rerun_nonce" not in st.session_state:
        st.session_state.rerun_nonce = 0
    if "log" not in st.session_state:
        st.session_state.log = []

    fleet = st.session_state.fleet

    # ---- Sidebar ----
    st.sidebar.title(":satellite: Fleet Controls")
    snr_db = st.sidebar.slider("Global SNR (dB)", 0.0, 20.0, 12.0, 0.5)
    n_scans = st.sidebar.slider("Mission scans", 10, 60, 30)
    show_gt = st.sidebar.toggle("Ground truth overlay", value=True)
    show_occl = st.sidebar.toggle("Occlusion shading", value=True)

    st.sidebar.markdown("---")
    st.sidebar.subheader("KWS engine")
    use_real = st.sidebar.toggle("Real NC-SSM/NC-TCN", value=False)
    kws_model = st.sidebar.selectbox("Model", ["nc-ssm", "nc-tcn"],
                                     disabled=not use_real)
    kws_backbone = st.sidebar.selectbox("Backbone", ["Tiny", "Small"],
                                        disabled=not use_real or
                                        kws_model != "nc-ssm")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Live hardware (selected drone only)")
    cam_idx = st.sidebar.number_input("Camera index", 0, 8, 0)
    mic_channels = st.sidebar.number_input("Mic channels", 1, 8, 4)
    mic_device = st.sidebar.number_input("Mic device index", 0, 32, 1)

    if st.sidebar.button("Re-run fleet sim", type="primary"):
        st.session_state.rerun_nonce += 1
        st.session_state.log.append(
            f"[{time.strftime('%H:%M:%S')}] fleet sim restarted")

    # ---- Build per-drone data ----
    fleet_data = {}
    for d in fleet:
        fleet_data[d["id"]] = cached_run(
            d["scenario"], snr_db, n_scans,
            "real" if use_real else "stub", kws_model, kws_backbone,
            d["id"], st.session_state.rerun_nonce)

    # ---- HEADER ----
    st.title(":helicopter: Drone Fleet Operations Console")
    st.caption(
        "Eyes (NC-Conv-SSM) + Ears (NC-SSM/NC-TCN) + "
        "Spatial (Mamba-COP-RFS GM-PHD)  |  total stack < 320 KB INT8  |  "
        "3.4 ms/scan on STM32H7 Cortex-M7")

    # ---- FLEET STATUS BAR ----
    cols = st.columns(len(fleet))
    for i, d in enumerate(fleet):
        sc, hist, lab = fleet_data[d["id"]]
        scan_idx = st.session_state.get("scan_idx", n_scans // 2)
        scan_idx = min(scan_idx, n_scans - 1)
        n_now = len(lab[scan_idx])
        n_total = sum(len(s) for s in lab)
        is_sel = (d["id"] == st.session_state.selected)
        with cols[i]:
            badge = "SELECTED" if is_sel else "select"
            label = (f":green[{d['id']}]" if is_sel
                     else d["id"])
            if st.button(f"{label}\n\nDetections: {n_now} now / "
                         f"{n_total} total\n\nScenario: {sc['name']}\n\n"
                         f":red[{badge}]" if is_sel else
                         f"{label}\n\nDetections: {n_now} now / "
                         f"{n_total} total\n\nScenario: {sc['name']}\n\n"
                         f"_(click to select)_",
                         key=f"sel_{d['id']}",
                         use_container_width=True):
                st.session_state.selected = d["id"]
                st.session_state.log.append(
                    f"[{time.strftime('%H:%M:%S')}] selected {d['id']}")

    selected = next(d for d in fleet
                     if d["id"] == st.session_state.selected)
    sel_sc, sel_hist, sel_lab = fleet_data[selected["id"]]

    st.markdown("---")

    # ---- TACTICAL MAP (left) | LIVE PANELS (right) ----
    map_col, live_col = st.columns([1.3, 1.0])

    with map_col:
        scan_idx = st.slider("Mission scan", 0, max(n_scans - 1, 0),
                             value=min(n_scans // 2, n_scans - 1),
                             key="scan_idx")
        st.pyplot(fig_tactical_map(fleet, selected["id"], fleet_data,
                                    scan_idx),
                  clear_figure=True)

    with live_col:
        st.markdown(f"### :round_pushpin: {selected['id']}  "
                    f"({sel_sc['name']})")
        if selected["live_cam"]:
            st.caption(":eye: Live webcam (this drone is the operator's)")
            if st.button("Capture camera frame",
                          key="eye_btn", use_container_width=True):
                try:
                    from drone_demo_vision import (load_ncconv_classifier,
                                                   classify, estimate_sigma)
                    if "vision_net" not in st.session_state:
                        net, status = load_ncconv_classifier()
                        st.session_state.vision_net = net
                        st.session_state.vision_status = status
                    pair = capture_camera(cam_idx)
                    if pair is None:
                        st.error(f"camera {cam_idx} unavailable")
                    else:
                        rgb, frame = pair
                        label, conf = classify(
                            st.session_state.vision_net, frame)
                        sigma = estimate_sigma(frame)
                        st.image(rgb, use_container_width=True,
                                 caption=f"NC-Conv: {label} "
                                         f"({conf*100:.0f}%) | "
                                         f"sigma {sigma:.2f}")
                        st.session_state.log.append(
                            f"[{time.strftime('%H:%M:%S')}] "
                            f"{selected['id']} EYE: {label}")
                except Exception as e:
                    st.error(f"Vision error: {e}")
        else:
            st.caption(":eye: (simulated -- this drone uses synthetic "
                       "vision)")
            st.info(f"Simulated NC-Conv on synthetic frame.\n"
                    f"In live deployment: drone-mounted camera.")

        if selected["live_mic"]:
            st.caption(":ear: Live mic (this drone is the operator's)")
            if st.button("Capture 1.5 s of audio",
                          key="ear_btn", use_container_width=True):
                try:
                    from drone_demo_kws import RealKWSClassifier
                    if ("kws_obj" not in st.session_state or
                        st.session_state.get("kws_tag") !=
                            (kws_model, kws_backbone)):
                        st.session_state.kws_obj = RealKWSClassifier(
                            model=kws_model, backbone=kws_backbone)
                        st.session_state.kws_tag = (kws_model,
                                                     kws_backbone)
                    rec = capture_mic(1.5, int(mic_channels),
                                       int(mic_device))
                    rms = np.sqrt((rec ** 2).mean(axis=0) + 1e-12)
                    mono = rec.mean(axis=1)
                    wav = np.pad(mono,
                                  (0, max(0, 16000 - len(mono))))[:16000]
                    label, conf = (
                        st.session_state.kws_obj.classify(wav)
                        if st.session_state.kws_obj.ok
                        else ("?", 0.0))
                    col = KW_COLOURS.get(label, "#7F8C8D")
                    st.markdown(
                        f"<div style='padding:10px;border-radius:8px;"
                        f"background:{col};color:white;text-align:center;"
                        f"font-size:1.4em;font-weight:700'>"
                        f"{label} <small>(conf {conf:.2f})</small></div>",
                        unsafe_allow_html=True)
                    st.caption(" ".join(f"ch{i}={r:.3f}"
                                         for i, r in enumerate(rms)))
                    st.session_state.log.append(
                        f"[{time.strftime('%H:%M:%S')}] "
                        f"{selected['id']} EAR: {label}")
                except Exception as e:
                    st.error(f"Mic error: {e}")
        else:
            st.caption(":ear: (simulated)")
            cur = sel_lab[min(scan_idx, len(sel_lab) - 1)]
            if cur:
                chips = []
                for _, _, kw in cur:
                    col = KW_COLOURS.get(kw, "#888")
                    chips.append(
                        "<span style='background:" + col +
                        ";color:white;padding:2px 8px;"
                        "border-radius:8px;font-size:0.85em'>" +
                        (kw or "?") + "</span>")
                st.markdown("Detected: " + "  ".join(chips),
                            unsafe_allow_html=True)
            else:
                st.info("no detections this scan")

        st.markdown("**Drone status**")
        s1, s2 = st.columns(2)
        s1.metric("Pos (x, y)",
                  f"{selected['pos'][0]}, {selected['pos'][1]}")
        s2.metric("Heading", f"{selected['heading']} deg")

    st.markdown("---")

    # ---- POLAR + TIMELINE ----
    p_col, t_col = st.columns([1.0, 1.6])
    with p_col:
        st.pyplot(fig_polar_for(sel_hist, sel_lab, scan_idx,
                                 selected["color"]),
                  clear_figure=True)
    with t_col:
        st.pyplot(fig_timeline(sel_hist, sel_lab, sel_sc, scan_idx,
                                selected["color"], show_gt, show_occl),
                  clear_figure=True)

    # ---- FLEET EVENT LOG ----
    st.markdown("---")
    st.subheader(":scroll: Fleet event log")
    if not st.session_state.log:
        st.caption("(no events yet -- click drones, capture cam/mic, etc.)")
    else:
        for line in reversed(st.session_state.log[-12:]):
            st.code(line, language=None)

    # ---- FOOTER ----
    st.markdown("---")
    st.caption(
        "Drone Fleet Operations Console -- IronDome-DOA-Tracking + NC family  |  "
        "github.com/DrJinHoChoi/IronDome-DOA-Tracking  |  "
        "Dual license (academic+commercial)  |  "
        "jinhochoi@smartear.co.kr")


if __name__ == "__main__":
    main()
