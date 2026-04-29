"""드론 함대 운용 콘솔 — IronDome Lattice (cyber ops UI).

NC family 통합:
    Eyes  : NC-Conv-SSM
    Ears  : NC-SSM / NC-TCN
    Spatial: Mamba-COP-RFS
"""
from __future__ import annotations

import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from drone_demo import (SCENARIOS, KW_COLOURS, run_pipeline,
                        attach_kws_labels)

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    def st_autorefresh(interval=1000, key=None, limit=None):
        return 0


st.set_page_config(page_title="IronDome 함대 콘솔",
                   page_icon=":helicopter:", layout="wide",
                   initial_sidebar_state="expanded")


# --------------------------------------------------------------------- #
# Cyber-ops CSS                                                          #
# --------------------------------------------------------------------- #
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Rajdhani:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

html, body, [class*="css"], .main {
    font-family: 'Rajdhani', 'Segoe UI', sans-serif !important;
}
.main { background: radial-gradient(ellipse at top, #0B1426 0%, #050810 70%) !important; }
.stApp { background: transparent !important; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #081222 0%, #050810 100%) !important;
    border-right: 1px solid rgba(0, 217, 255, 0.18);
}
section[data-testid="stSidebar"] * { font-family: 'Rajdhani', sans-serif !important; }

/* Hero title */
.hero {
    font-family: 'Orbitron', monospace !important;
    font-weight: 900;
    font-size: 1.9rem;
    letter-spacing: 0.06em;
    background: linear-gradient(90deg, #00D9FF 0%, #7C5CFF 50%, #FF00B7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 0 40px rgba(0, 217, 255, 0.18);
    margin: 0;
    padding: 0;
}
.hero-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #5B7FA3;
    letter-spacing: 0.18em;
    text-transform: uppercase;
}

/* Status strip */
.status-strip {
    display: flex;
    gap: 14px;
    padding: 12px 14px;
    background: rgba(8, 18, 34, 0.85);
    border: 1px solid rgba(0, 217, 255, 0.20);
    border-radius: 6px;
    backdrop-filter: blur(8px);
    margin-bottom: 12px;
}
.subsys {
    flex: 1;
    text-align: center;
    border-right: 1px solid rgba(0, 217, 255, 0.10);
    padding: 4px 6px;
}
.subsys:last-child { border-right: none; }
.subsys-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.66rem;
    color: #5B7FA3;
    letter-spacing: 0.14em;
    text-transform: uppercase;
}
.subsys-value {
    font-family: 'Orbitron', monospace;
    font-size: 1.05rem;
    font-weight: 700;
    color: #00D9FF;
}
.subsys-value.warn { color: #FFB400; }
.subsys-value.alert { color: #FF3B5C; text-shadow: 0 0 8px rgba(255,59,92,0.6); }
.subsys-value.ok { color: #2EE6A6; }

.led {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 6px;
    box-shadow: 0 0 8px currentColor;
    animation: pulse 1.6s infinite ease-in-out;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.55; transform: scale(0.85); }
}
.led.green  { background: #2EE6A6; color: #2EE6A6; }
.led.amber  { background: #FFB400; color: #FFB400; }
.led.red    { background: #FF3B5C; color: #FF3B5C; }

/* Fleet card */
.unit-card {
    background: linear-gradient(135deg, rgba(15, 30, 56, 0.95) 0%, rgba(8, 18, 34, 0.95) 100%);
    border: 1px solid rgba(0, 217, 255, 0.20);
    border-radius: 8px;
    padding: 14px;
    color: #E5F1FF;
    font-family: 'Rajdhani', sans-serif;
    transition: all 0.2s;
    position: relative;
}
.unit-card.selected {
    border: 1px solid #00D9FF;
    box-shadow: 0 0 24px rgba(0, 217, 255, 0.32),
                inset 0 0 12px rgba(0, 217, 255, 0.08);
}
.unit-card.selected::before {
    content: "ACTIVE";
    position: absolute;
    top: -10px; left: 14px;
    background: #00D9FF;
    color: #0B1426;
    padding: 2px 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    border-radius: 3px;
}
.unit-id {
    font-family: 'Orbitron', monospace;
    font-weight: 700;
    font-size: 1.05rem;
    letter-spacing: 0.04em;
}
.unit-role {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #5B7FA3;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.unit-line {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #B7CDE6;
    line-height: 1.55;
}
.unit-line .k { color: #5B7FA3; }
.unit-line .v { color: #E5F1FF; font-weight: 600; }
.unit-line .v.warn { color: #FFB400; }
.unit-line .v.alert { color: #FF3B5C; }
.unit-line .v.ok { color: #2EE6A6; }
.unit-bar {
    height: 4px;
    background: rgba(0, 217, 255, 0.1);
    border-radius: 2px;
    margin: 6px 0;
    overflow: hidden;
}
.unit-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #2EE6A6, #00D9FF);
    box-shadow: 0 0 6px rgba(0, 217, 255, 0.6);
}

/* Threat row */
.threat-row {
    background: linear-gradient(90deg, rgba(255, 59, 92, 0.10) 0%, rgba(8, 18, 34, 0.6) 60%);
    border-left: 3px solid #FF3B5C;
    padding: 8px 12px;
    margin: 4px 0;
    color: #E5F1FF;
    border-radius: 3px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
}
.threat-row.lvl3 { border-left-color: #FFB400; background: linear-gradient(90deg, rgba(255, 180, 0, 0.10) 0%, rgba(8, 18, 34, 0.6) 60%); }
.threat-row.lvl2 { border-left-color: #00D9FF; background: linear-gradient(90deg, rgba(0, 217, 255, 0.08) 0%, rgba(8, 18, 34, 0.6) 60%); }
.threat-row.lvl1 { border-left-color: #5B7FA3; background: rgba(8, 18, 34, 0.6); }

.threat-lvl {
    display: inline-block;
    padding: 1px 8px;
    border-radius: 3px;
    font-weight: 700;
    margin-right: 8px;
}
.threat-lvl.l5 { background: #FF3B5C; color: white; }
.threat-lvl.l4 { background: #FF7A1A; color: white; }
.threat-lvl.l3 { background: #FFB400; color: #0B1426; }
.threat-lvl.l2 { background: #00D9FF; color: #0B1426; }
.threat-lvl.l1 { background: #5B7FA3; color: white; }

/* Section titles */
.sec-title {
    font-family: 'Orbitron', monospace !important;
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    color: #00D9FF !important;
    border-bottom: 1px solid rgba(0, 217, 255, 0.18);
    padding-bottom: 4px;
    margin: 18px 0 10px 0 !important;
}

/* Alert banner */
.alert-band {
    background: linear-gradient(90deg, rgba(255, 59, 92, 0.30) 0%, rgba(255, 59, 92, 0.05) 80%);
    border: 1px solid rgba(255, 59, 92, 0.5);
    color: #FF3B5C;
    padding: 8px 14px;
    border-radius: 4px;
    font-family: 'Orbitron', monospace;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    animation: bandpulse 2s infinite ease-in-out;
}
.alert-band.warn {
    background: linear-gradient(90deg, rgba(255, 180, 0, 0.25) 0%, rgba(255, 180, 0, 0.05) 80%);
    border-color: rgba(255, 180, 0, 0.5);
    color: #FFB400;
    animation: none;
}
@keyframes bandpulse {
    0%, 100% { box-shadow: 0 0 0 rgba(255, 59, 92, 0.4); }
    50%      { box-shadow: 0 0 22px rgba(255, 59, 92, 0.6); }
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #0F2440 0%, #081A30 100%);
    border: 1px solid rgba(0, 217, 255, 0.32);
    color: #00D9FF;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    transition: all 0.18s;
}
.stButton > button:hover {
    border-color: #00D9FF;
    color: #ffffff;
    box-shadow: 0 0 16px rgba(0, 217, 255, 0.32);
    background: linear-gradient(135deg, #0F2440 0%, #1A3560 100%);
}
.stButton > button:focus { box-shadow: 0 0 16px rgba(0, 217, 255, 0.32); }
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #00D9FF 0%, #7C5CFF 100%);
    color: #0B1426;
    border: none;
    font-weight: 700;
}

/* Slider */
.stSlider > div > div > div { background: rgba(0, 217, 255, 0.2) !important; }

/* Code block */
code, .stCodeBlock {
    background: rgba(8, 18, 34, 0.85) !important;
    color: #00D9FF !important;
    font-family: 'JetBrains Mono', monospace !important;
    border: 1px solid rgba(0, 217, 255, 0.18) !important;
    border-radius: 4px !important;
}

/* Expanders */
[data-testid="stExpander"] {
    background: rgba(8, 18, 34, 0.6) !important;
    border: 1px solid rgba(0, 217, 255, 0.18) !important;
    border-radius: 4px !important;
}

/* Hide default Streamlit chrome */
header[data-testid="stHeader"] { background: transparent !important; }
.block-container { padding-top: 1.2rem !important; }
footer { visibility: hidden; }

/* Text overrides */
h1, h2, h3, h4 { font-family: 'Orbitron', monospace !important;
                  color: #E5F1FF !important; }
p, label, span, div { color: #E5F1FF; }
.stMarkdown { color: #E5F1FF; }
[data-testid="stMetricLabel"] {
    font-family: 'JetBrains Mono', monospace !important;
    color: #5B7FA3 !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}
[data-testid="stMetricValue"] {
    font-family: 'Orbitron', monospace !important;
    color: #00D9FF !important;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------- #
# 함대 정의                                                              #
# --------------------------------------------------------------------- #
def default_fleet():
    return [
        {"id": "ALPHA-01", "code": "A1", "pos": (-30, 20), "heading": 30,
         "scenario": "sar", "live_cam": True, "live_mic": True,
         "color": "#00D9FF", "battery": 87, "alt": 120, "speed": 12,
         "role": "OPERATOR / SAR LEAD"},
        {"id": "ALPHA-02", "code": "A2", "pos": (15, 35),  "heading": 270,
         "scenario": "sar", "live_cam": False, "live_mic": False,
         "color": "#2EE6A6", "battery": 73, "alt": 95, "speed": 14,
         "role": "SAR RECON"},
        {"id": "BRAVO-01", "code": "B1", "pos": (40, -10), "heading": 200,
         "scenario": "combat", "live_cam": False, "live_mic": False,
         "color": "#FFB400", "battery": 64, "alt": 200, "speed": 18,
         "role": "TACTICAL RECON"},
        {"id": "BRAVO-02", "code": "B2", "pos": (-15, -25), "heading": 110,
         "scenario": "combat", "live_cam": False, "live_mic": False,
         "color": "#FF3B5C", "battery": 41, "alt": 180, "speed": 20,
         "role": "TACTICAL RECON"},
    ]


KW_KR = {
    "help": "구조요청", "water": "물/식수", "down": "아래쪽",
    "here": "여기있음", "vehicle": "차량", "drone": "드론음",
    "shot": "총성", "yes": "예", "no": "아니오", "up": "위쪽",
    "left": "좌측", "right": "우측", "on": "켜짐", "off": "꺼짐",
    "stop": "정지", "go": "출발", "silence": "정적",
    "unknown": "미분류",
}


def kw_kr(kw):
    return KW_KR.get(kw, kw or "?")


def threat_level(kw):
    t = {"help": 5, "shot": 5, "vehicle": 4, "drone": 4,
         "down": 3, "water": 2, "here": 3}
    return t.get(kw, 1)


def battery_class(p):
    return "ok" if p > 60 else "warn" if p > 30 else "alert"


# --------------------------------------------------------------------- #
# 시뮬 캐시                                                              #
# --------------------------------------------------------------------- #
def _scenario_with_overrides(key, snr_db, n_scans):
    base = deepcopy(SCENARIOS[key])
    base["snr_db"] = float(snr_db)
    base["n_scans"] = int(n_scans)
    base["sources"] = [(min(t_in, n_scans - 1), min(t_out, n_scans),
                        th0, th1, kw)
                       for t_in, t_out, th0, th1, kw in base["sources"]]
    base["occlusions"] = [(idx, min(a, n_scans), min(b, n_scans))
                          for idx, a, b in base["occlusions"]]
    return base


@st.cache_data(show_spinner="공간 인식 파이프라인 가동...")
def cached_run(key, snr_db, n_scans, kws_mode, kws_model,
               kws_backbone, drone_id, nonce=0):
    sc = _scenario_with_overrides(key, snr_db, n_scans)
    history = run_pipeline(sc)
    labelled = attach_kws_labels(history, sc,
                                 real_kws=(kws_mode == "real"),
                                 model=kws_model, backbone=kws_backbone)
    return sc, history, labelled


# --------------------------------------------------------------------- #
# 시각화                                                                 #
# --------------------------------------------------------------------- #
def fig_tactical(fleet, selected_id, fleet_data, scan_idx):
    fig, ax = plt.subplots(figsize=(8.5, 7))
    bg = "#050810"
    ax.set_facecolor(bg)
    fig.patch.set_facecolor(bg)
    ax.set_xlim(-65, 65); ax.set_ylim(-55, 55)
    ax.set_aspect("equal")

    # 격자
    for r in (10, 20, 30, 40, 50):
        circ = plt.Circle((0, 0), r, fill=False, color="#1A3A5F",
                          lw=0.6, alpha=0.55, linestyle="--")
        ax.add_patch(circ)
        ax.text(r * 0.71, -r * 0.71, f"{r:02d} KM", color="#1A3A5F",
                fontsize=7, ha="center",
                family="monospace")
    ax.axhline(0, color="#1A3A5F", lw=0.4, alpha=0.45)
    ax.axvline(0, color="#1A3A5F", lw=0.4, alpha=0.45)
    # Compass
    for ang, lab in [(0, "N"), (90, "E"), (180, "S"), (270, "W")]:
        x = 56 * np.sin(np.deg2rad(ang))
        y = 56 * np.cos(np.deg2rad(ang))
        ax.text(x, y, lab, color="#00D9FF", fontsize=14,
                fontweight="bold", ha="center", va="center",
                family="monospace")

    # Crosshair center
    ax.plot([0], [0], "+", color="#00D9FF", ms=20, mew=1.2, alpha=0.8)

    # 표적
    target_xy = []
    for d in fleet:
        sc, hist, lab = fleet_data[d["id"]]
        if scan_idx >= len(lab):
            continue
        dx, dy = d["pos"]
        for tid, deg, kw in lab[scan_idx]:
            r = 25
            theta_world = np.deg2rad(d["heading"] + deg)
            tx = dx + r * np.cos(theta_world)
            ty = dy + r * np.sin(theta_world)
            target_xy.append((tx, ty, kw))
            ax.plot([dx, tx], [dy, ty],
                    color=KW_COLOURS.get(kw, "#888"),
                    lw=0.8, alpha=0.45, linestyle=":")

    for x, y, kw in target_xy:
        col = KW_COLOURS.get(kw, "#888")
        lvl = threat_level(kw)
        size = 70 + lvl * 28
        ax.scatter([x], [y], s=size, color=col, marker="X",
                   edgecolors="white", linewidths=1.4, zorder=5,
                   alpha=0.95)
        if lvl >= 4:
            for r_ring, alpha in [(3, 0.7), (4.5, 0.4)]:
                ring = plt.Circle((x, y), r_ring, fill=False,
                                   color="#FF3B5C", lw=1.5, alpha=alpha)
                ax.add_patch(ring)

    # Drones
    for d in fleet:
        x, y = d["pos"]
        is_sel = (d["id"] == selected_id)
        theta = np.deg2rad(d["heading"])
        # FOV cone
        for sweep in np.linspace(-np.pi / 2, np.pi / 2, 14):
            ax.plot([x, x + 28 * np.cos(theta + sweep)],
                    [y, y + 28 * np.sin(theta + sweep)],
                    color=d["color"], lw=0.4, alpha=0.10)
        # Triangle
        size = 4
        tri = np.array([[size, 0], [-size * 0.6, size * 0.6],
                         [-size * 0.6, -size * 0.6]])
        c, s = np.cos(theta), np.sin(theta)
        rot = np.array([[c, -s], [s, c]])
        tri = (rot @ tri.T).T + np.array([x, y])
        poly = plt.Polygon(tri, color=d["color"],
                            ec="#FFFFFF" if is_sel else "white",
                            lw=2 if is_sel else 0.6, zorder=6)
        ax.add_patch(poly)
        ax.text(x, y - 7, d["code"], color=d["color"],
                fontsize=9, fontweight="bold", ha="center",
                family="monospace",
                bbox=dict(facecolor=bg, edgecolor="none",
                          alpha=0.85, pad=2))
        if is_sel:
            for r in (6, 8.5):
                ring = plt.Circle((x, y), r, fill=False,
                                   color="#00D9FF", lw=1.0,
                                   alpha=0.7 - r * 0.05,
                                   linestyle="-")
                ax.add_patch(ring)

    ax.set_xticks([]); ax.set_yticks([])
    ax.text(-62, 51, f"TACTICAL  //  T+{scan_idx:03d}",
            color="#00D9FF", fontsize=11, fontweight="bold",
            family="monospace")
    ax.text(-62, 47, f"FLEET={len(fleet):02d}  CONTACTS={len(target_xy):02d}",
            color="#5B7FA3", fontsize=9, family="monospace")
    ax.text(45, 51, f"GRID 10KM", color="#5B7FA3",
            fontsize=8, family="monospace")
    for spine in ax.spines.values():
        spine.set_color("#1A3A5F")
    fig.tight_layout()
    return fig


def fig_polar(history, labelled, scan_idx, drone_color):
    angles = history["scan_angles_deg"]
    P = history["spectra"][scan_idx]
    P = P / (P.max() + 1e-9)
    fig = plt.figure(figsize=(5.4, 4.6))
    fig.patch.set_facecolor("#050810")
    ax = fig.add_subplot(111, projection="polar")
    ax.set_facecolor("#0B1426")
    ax.plot(np.deg2rad(angles), P, color=drone_color, lw=1.6)
    ax.fill(np.deg2rad(angles), P, color=drone_color, alpha=0.15)
    for tid, deg, kw in labelled[scan_idx]:
        col = KW_COLOURS.get(kw, "#888")
        ax.plot([np.deg2rad(deg)], [1.0], "o", color=col, ms=12,
                markeredgecolor="white", markeredgewidth=1.5)
        ax.text(np.deg2rad(deg), 1.22, kw_kr(kw), color=col,
                fontweight="bold", fontsize=9, ha="center",
                family="monospace")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetamin(-90); ax.set_thetamax(90)
    ax.set_yticklabels([])
    ax.tick_params(colors="#5B7FA3")
    ax.set_title("DOA SPECTRUM // 8-CH ULA",
                 color="#00D9FF", fontsize=10, fontweight="bold",
                 family="monospace", pad=10)
    fig.tight_layout()
    return fig


def fig_timeline(history, labelled, scenario, scan_idx, color,
                 show_gt, show_occl):
    n = scenario["n_scans"]
    angles = history["scan_angles_deg"]
    fig, ax = plt.subplots(figsize=(18, 5.0))
    fig.patch.set_facecolor("#050810")
    ax.set_facecolor("#0B1426")
    palette = plt.cm.tab10(np.linspace(0, 1, 10))
    by_id = {}
    for s, per in enumerate(labelled):
        for tid, deg, kw in per:
            by_id.setdefault(tid, []).append((s, deg, kw))
    for tid, pts in by_id.items():
        ss = [p[0] for p in pts]; ds = [p[1] for p in pts]
        ax.plot(ss, ds, "-", color=palette[tid % 10], lw=1.7,
                marker="o", ms=3)
        prev = None
        for s, d, kw in pts:
            if kw and kw != prev:
                col = KW_COLOURS.get(kw, "#888")
                ax.text(s, d + 6, kw_kr(kw), fontsize=7, color=col,
                        fontweight="bold", ha="center",
                        family="monospace")
                prev = kw
    if show_gt:
        gt = history["gt"]
        for k in range(gt.shape[1]):
            ax.plot(np.arange(n), gt[:, k], color="#5B7FA3",
                    lw=0.5, linestyle="--", alpha=0.4)
    if show_occl:
        for sidx, a, b in scenario["occlusions"]:
            ax.axvspan(a, b, alpha=0.18, color="#FFB400")
    ax.axvline(scan_idx, color=color, lw=2, alpha=0.9)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(angles[0], angles[-1])
    ax.set_xlabel("MISSION SCAN", color="#5B7FA3", family="monospace")
    ax.set_ylabel("BEARING (DEG)", color="#5B7FA3", family="monospace")
    ax.tick_params(colors="#5B7FA3")
    for spine in ax.spines.values():
        spine.set_color("#1A3A5F")
    ax.grid(alpha=0.18, color="#1A3A5F")
    ax.set_title("TRACK + KWS TIMELINE",
                 color="#00D9FF", fontsize=10, fontweight="bold",
                 family="monospace", loc="left")
    fig.tight_layout()
    return fig


def fig_sparkline(values, color="#00D9FF"):
    fig, ax = plt.subplots(figsize=(2.5, 0.6))
    fig.patch.set_facecolor("#0B1426")
    ax.set_facecolor("#0B1426")
    if len(values) > 1:
        ax.plot(values, color=color, lw=1.5)
        ax.fill_between(range(len(values)), values, color=color, alpha=0.18)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlim(0, max(len(values) - 1, 1))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig


# --------------------------------------------------------------------- #
# 라이브 캡처                                                            #
# --------------------------------------------------------------------- #
_CAM_HANDLES = {}   # cache cv2.VideoCapture across captures (avoid open/close)


def _get_cam(idx):
    import cv2
    idx = int(idx)
    if idx not in _CAM_HANDLES or not _CAM_HANDLES[idx].isOpened():
        # Use DSHOW backend on Windows -- much faster init than default
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(idx)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # latest frame only
        _CAM_HANDLES[idx] = cap
    return _CAM_HANDLES[idx]


def capture_camera(idx):
    """Grab one frame from a cached camera handle (no open/close each time)."""
    import cv2
    cam = _get_cam(idx)
    if not cam.isOpened():
        return None
    # Drain any stale buffered frame to keep latency low
    for _ in range(2):
        cam.grab()
    ok, frame = cam.read()
    if not ok or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), frame


def capture_mic(seconds, channels, device):
    """Non-blocking-ish: record then return. Default 0.5 s for auto mode
    to avoid UI freeze (was 1.5 s)."""
    import sounddevice as sd
    n = int(seconds * 16000)
    rec = sd.rec(n, samplerate=16000, channels=channels,
                 device=device, dtype="float32", blocking=False)
    sd.wait()
    return rec


# --------------------------------------------------------------------- #
# 페이지                                                                 #
# --------------------------------------------------------------------- #
def main():
    if "fleet" not in st.session_state:
        st.session_state.fleet = default_fleet()
    if "selected" not in st.session_state:
        st.session_state.selected = st.session_state.fleet[0]["id"]
    if "rerun_nonce" not in st.session_state:
        st.session_state.rerun_nonce = 0
    if "log" not in st.session_state:
        st.session_state.log = [
            f"[{time.strftime('%H:%M:%S')}] SYSTEM ONLINE — FLEET 4/4 READY"
        ]
    if "mission_start" not in st.session_state:
        st.session_state.mission_start = time.time()
    if "threat_history" not in st.session_state:
        st.session_state.threat_history = []

    fleet = st.session_state.fleet

    # ---- Sidebar ----
    st.sidebar.markdown(
        "<div style='font-family:Orbitron,monospace;font-size:1.1rem;"
        "font-weight:700;color:#00D9FF;letter-spacing:0.1em'>"
        "OPS CONTROL</div><hr style='border-color:rgba(0,217,255,0.2)'>",
        unsafe_allow_html=True)

    st.sidebar.markdown("**MISSION ENV**")
    snr_db = st.sidebar.slider("ACOUSTIC SNR (dB)", 0.0, 20.0, 12.0, 0.5)
    n_scans = st.sidebar.slider("MISSION DURATION (scans)", 10, 60, 30)
    show_gt = st.sidebar.toggle("Ground truth overlay", value=False)
    show_occl = st.sidebar.toggle("Occlusion shading", value=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**KWS ENGINE**")
    use_real = st.sidebar.toggle("Real NC-SSM/NC-TCN", value=False)
    kws_model = st.sidebar.selectbox("Model", ["nc-ssm", "nc-tcn"],
                                     disabled=not use_real)
    kws_backbone = st.sidebar.selectbox("Backbone", ["Tiny", "Small"],
                                        disabled=not use_real
                                        or kws_model != "nc-ssm")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**OPERATOR HW**")
    cam_idx = st.sidebar.number_input("Camera idx", 0, 8, 0)
    mic_channels = st.sidebar.number_input("Mic channels", 1, 8, 4)
    mic_device = st.sidebar.number_input("Mic device idx", 0, 32, 1)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**:arrow_forward: AUTONOMOUS**")
    auto_play = st.sidebar.toggle("Auto-advance scan", value=False)
    play_speed = st.sidebar.select_slider("Speed",
        options=[0.5, 1.0, 1.5, 2.0, 3.0], value=1.0,
        format_func=lambda v: f"{v}x", disabled=not auto_play)
    auto_capture_cam = st.sidebar.toggle("Auto camera (3s)", value=False)
    auto_capture_mic = st.sidebar.toggle("Auto mic (3s)", value=False)
    auto_select = st.sidebar.toggle("Cycle drones (10s)", value=False)

    st.sidebar.markdown("---")
    if st.sidebar.button(":arrows_counterclockwise:  RESTART MISSION",
                         type="primary", use_container_width=True):
        st.session_state.rerun_nonce += 1
        st.session_state.mission_start = time.time()
        st.session_state.log.append(
            f"[{time.strftime('%H:%M:%S')}] MISSION RESET")
        st.session_state.threat_history = []

    # ---- Fleet data ----
    fleet_data = {}
    for d in fleet:
        fleet_data[d["id"]] = cached_run(
            d["scenario"], snr_db, n_scans,
            "real" if use_real else "stub", kws_model, kws_backbone,
            d["id"], st.session_state.rerun_nonce)

    # ---- Auto refresh ----
    # 자동 재생만 켜면 빠른 tick (1s/play_speed); 캡처가 켜진 경우엔
    # blocking I/O 동안 UI 멈춤을 줄이려고 인터벌을 늘림.
    if auto_play or auto_capture_cam or auto_capture_mic or auto_select:
        if auto_capture_mic:
            interval_ms = 1500       # 마이크 0.5s + 처리 여유
        elif auto_capture_cam:
            interval_ms = 1000       # 카메라 grab은 빠름
        elif auto_play:
            interval_ms = max(500, int(1000 / play_speed))
        else:
            interval_ms = 1000
        tick = st_autorefresh(interval=interval_ms, key="ops_tick")
    else:
        tick = 0

    if auto_play:
        cur_scan = st.session_state.get("scan_idx", 0)
        st.session_state.scan_idx = (cur_scan + 1) % n_scans

    if auto_select:
        if "auto_select_last" not in st.session_state:
            st.session_state.auto_select_last = time.time()
        if time.time() - st.session_state.auto_select_last > 10:
            ids = [d["id"] for d in fleet]
            cur_i = ids.index(st.session_state.selected)
            next_i = (cur_i + 1) % len(ids)
            st.session_state.selected = ids[next_i]
            st.session_state.auto_select_last = time.time()
            st.session_state.log.append(
                f"[{time.strftime('%H:%M:%S')}] CYCLE -> {ids[next_i]}")

    scan_idx = min(st.session_state.get("scan_idx", n_scans // 2),
                   n_scans - 1)

    # ---- Onboarding (first visit, dismissible) ----
    if "onboard_dismissed" not in st.session_state:
        st.session_state.onboard_dismissed = False
    if not st.session_state.onboard_dismissed:
        with st.container():
            st.markdown(
                "<div style='background:linear-gradient(135deg,#0F2545,"
                "#1A3A5F);border:2px solid #00D9FF;border-radius:8px;"
                "padding:18px 22px;margin-bottom:14px;color:#E5F0FF;"
                "font-family:Consolas,monospace'>"
                "<div style='color:#00D9FF;font-size:1.3em;"
                "font-weight:700;margin-bottom:10px'>"
                ":eyes:  처음 방문하셨나요? 30초 안에 데모 이해하기"
                "</div>"
                "<div style='display:grid;grid-template-columns:1fr 1fr;"
                "gap:18px;font-size:0.92em;line-height:1.55'>"

                # Left column
                "<div>"
                "<b style='color:#FFD60A'>:helicopter: 무엇을 보여주는가</b><br>"
                "4기 드론 함대를 가상 시나리오로 운용하는 콘솔. "
                "논문이 제안한 4종 AI 모델 (NC-Conv-SSM 영상, "
                "NC-SSM/NC-TCN 음성, Mamba-COP-RFS 공간) 이 "
                "<b>실시간 처리한 결과</b>를 시각화.<br><br>"
                "<b style='color:#2EE6A6'>:white_check_mark: 진짜 vs 시뮬</b><br>"
                "&bull; 알파-1 (운용자기 :star:): 노트북 진짜 카메라/마이크<br>"
                "&bull; 알파-2 / 브라보-1,2: 시뮬레이션<br>"
                "&bull; 8-mic 어레이 신호: 시뮬 (실배치 시 ReSpeaker)<br>"
                "&bull; 드론 위치/배터리: 데모용 가짜 값"
                "</div>"

                # Right column
                "<div>"
                "<b style='color:#FFD60A'>:bar_chart: 패널별 데이터</b><br>"
                "<b>① 헤더</b>: 임무 시간, 위협 점수, 모델 풋프린트<br>"
                "<b>② 함대 카드</b>: 4기 상태 (배터리/임무/탐지수)<br>"
                "<b>③ 전술 상황도</b>: 드론 위치 + 표적 X 마크<br>"
                "<b>④ 운용자기 상세</b>: 라이브 카메라/마이크<br>"
                "<b>⑤ 위협 보드</b>: 우선순위 정렬 (위협도 1-5)<br>"
                "<b>⑥ 폴라 스펙트럼</b>: 음원 방향 분포 (현재 스캔)<br>"
                "<b>⑦ 추적 시간선</b>: 표적 30스캔 궤적 + KWS 라벨<br><br>"
                "<b style='color:#FF3B5C'>:rocket: 자동 데모</b><br>"
                "사이드바 <i>실시간 운용</i>의 토글 4개를 켜면 손 떼고 진행 가능."
                "</div>"
                "</div></div>",
                unsafe_allow_html=True)
            cb1, cb2, cb3 = st.columns([1, 1, 4])
            with cb1:
                if st.button(":white_check_mark: 이해했습니다",
                              use_container_width=True, type="primary"):
                    st.session_state.onboard_dismissed = True
                    st.rerun()
            with cb2:
                if st.button(":books: 다시 안 보기",
                              use_container_width=True):
                    st.session_state.onboard_dismissed = True
                    st.rerun()
    else:
        # 작은 도움말 토글 (사이드바)
        if st.sidebar.button(":eyes: 데모 가이드 다시 보기",
                              use_container_width=True):
            st.session_state.onboard_dismissed = False
            st.rerun()

    # ---- Header HERO ----
    elapsed = int(time.time() - st.session_state.mission_start)
    h, m, s = elapsed // 3600, (elapsed // 60) % 60, elapsed % 60

    total_threats = sum(
        sum(threat_level(kw) for _, _, kw in lab[min(scan_idx, len(lab) - 1)])
        for sc, hist, lab in fleet_data.values())
    st.session_state.threat_history.append(total_threats)
    if len(st.session_state.threat_history) > 60:
        st.session_state.threat_history = (
            st.session_state.threat_history[-60:])
    threat_class = ("alert" if total_threats >= 15
                    else "warn" if total_threats >= 8 else "ok")

    st.markdown(
        "<div class='hero'>IRONDOME // FLEET OPERATIONS CONSOLE</div>"
        "<div class='hero-sub'>NC-SSM • NC-TCN • Mamba-COP-RFS • "
        "NC-Conv-SSM &nbsp;|&nbsp; EDGE STACK &lt; 320 KB INT8 &nbsp;|&nbsp; "
        "3.4 ms / scan</div>",
        unsafe_allow_html=True)

    # Subsystem strip
    led_kws = ("green" if use_real else "amber")
    led_threat = ("red" if threat_class == "alert"
                  else "amber" if threat_class == "warn" else "green")
    avg_battery = int(np.mean([d["battery"] for d in fleet]))
    led_bat = ("green" if avg_battery > 60 else
               "amber" if avg_battery > 30 else "red")

    strip = f"""
    <div class='status-strip'>
      <div class='subsys'>
        <div class='subsys-label'><span class='led green'></span>MISSION CLOCK</div>
        <div class='subsys-value'>{h:02d}:{m:02d}:{s:02d}</div>
      </div>
      <div class='subsys'>
        <div class='subsys-label'><span class='led green'></span>FLEET</div>
        <div class='subsys-value ok'>{len(fleet):02d} / {len(fleet):02d}</div>
      </div>
      <div class='subsys'>
        <div class='subsys-label'><span class='led {led_bat}'></span>AVG BATTERY</div>
        <div class='subsys-value {battery_class(avg_battery)}'>{avg_battery}%</div>
      </div>
      <div class='subsys'>
        <div class='subsys-label'><span class='led {led_threat}'></span>THREAT LEVEL</div>
        <div class='subsys-value {threat_class}'>{total_threats:02d}</div>
      </div>
      <div class='subsys'>
        <div class='subsys-label'><span class='led {led_kws}'></span>KWS</div>
        <div class='subsys-value'>{'REAL' if use_real else 'SIM'}</div>
      </div>
      <div class='subsys'>
        <div class='subsys-label'><span class='led green'></span>SCAN</div>
        <div class='subsys-value'>{scan_idx + 1:03d} / {n_scans:03d}</div>
      </div>
      <div class='subsys'>
        <div class='subsys-label'><span class='led green'></span>LATENCY</div>
        <div class='subsys-value'>3.4 MS</div>
      </div>
    </div>
    """
    st.markdown(strip, unsafe_allow_html=True)

    # Alert banner
    if total_threats >= 15:
        st.markdown(
            f"<div class='alert-band'>:rotating_light:  CRITICAL — "
            f"MULTIPLE HIGH-THREAT CONTACTS &nbsp;|&nbsp; SCORE {total_threats}</div>",
            unsafe_allow_html=True)
    elif total_threats >= 8:
        st.markdown(
            f"<div class='alert-band warn'>:warning:  ELEVATED — "
            f"MONITORING ACTIVE &nbsp;|&nbsp; SCORE {total_threats}</div>",
            unsafe_allow_html=True)

    # ---- Fleet roster ----
    st.markdown("<div class='sec-title'>:helicopter:  FLEET ROSTER</div>",
                unsafe_allow_html=True)
    cols = st.columns(len(fleet))
    for i, d in enumerate(fleet):
        sc, hist, lab = fleet_data[d["id"]]
        n_now = len(lab[min(scan_idx, len(lab) - 1)])
        n_total = sum(len(s) for s in lab)
        is_sel = (d["id"] == st.session_state.selected)
        bcls = battery_class(d["battery"])
        sel_cls = "selected" if is_sel else ""

        with cols[i]:
            html = f"""
            <div class='unit-card {sel_cls}' style='border-left:4px solid {d["color"]}'>
              <div class='unit-id' style='color:{d["color"]}'>{d["id"]}</div>
              <div class='unit-role'>{d["role"]}</div>
              <div class='unit-line'>
                <span class='k'>BAT</span>  <span class='v {bcls}'>{d["battery"]}%</span>
              </div>
              <div class='unit-bar'><div class='unit-bar-fill' style='width:{d["battery"]}%'></div></div>
              <div class='unit-line'>
                <span class='k'>ALT</span>  <span class='v'>{d["alt"]:>3d} m</span> &nbsp;
                <span class='k'>SPD</span>  <span class='v'>{d["speed"]:>2d} m/s</span>
              </div>
              <div class='unit-line'>
                <span class='k'>HDG</span>  <span class='v'>{d["heading"]:03d}°</span> &nbsp;
                <span class='k'>POS</span>  <span class='v'>({d["pos"][0]:+03d},{d["pos"][1]:+03d})</span>
              </div>
              <div class='unit-line'>
                <span class='k'>DET</span>  <span class='v ok'>{n_now}</span> /
                <span class='v'>{n_total}</span>
              </div>
              <div class='unit-line'>
                <span class='k'>OPS</span>  <span class='v' style='color:{d["color"]}'>{sc["name"].upper()}</span>
              </div>
            </div>
            """
            st.markdown(html, unsafe_allow_html=True)
            if st.button(("◉ SELECTED" if is_sel else "○ SELECT"),
                         key=f"sel_{d['id']}",
                         use_container_width=True,
                         disabled=is_sel):
                st.session_state.selected = d["id"]
                st.session_state.log.append(
                    f"[{time.strftime('%H:%M:%S')}] {d['id']} SELECTED")

    selected = next(d for d in fleet
                     if d["id"] == st.session_state.selected)
    sel_sc, sel_hist, sel_lab = fleet_data[selected["id"]]

    # ---- TACTICAL + LIVE ----
    st.markdown("<div class='sec-title'>:compass:  TACTICAL OVERVIEW</div>",
                unsafe_allow_html=True)
    map_col, live_col = st.columns([1.4, 1.0])

    with map_col:
        if auto_play:
            st.progress((scan_idx + 1) / n_scans,
                        text=f"AUTO-ADVANCE  //  T+{scan_idx + 1:03d} / {n_scans:03d}")
        else:
            st.session_state.scan_idx = st.slider(
                "MISSION SCAN", 0, max(n_scans - 1, 0),
                value=scan_idx, key="scan_slider")
            scan_idx = st.session_state.scan_idx
        st.pyplot(fig_tactical(fleet, selected["id"], fleet_data,
                                scan_idx),
                  clear_figure=True)

    with live_col:
        st.markdown(
            f"<div class='sec-title' style='margin-top:0'>"
            f":satellite_antenna:  {selected['id']} LIVE FEED</div>",
            unsafe_allow_html=True)
        st.markdown(
            f"<span class='led green'></span><span style='font-family:JetBrains Mono,monospace;color:#5B7FA3;font-size:0.78rem'>"
            f"OPS={sel_sc['name'].upper()} &nbsp;|&nbsp; "
            f"POS=({selected['pos'][0]:+03d},{selected['pos'][1]:+03d}) &nbsp;|&nbsp; "
            f"HDG={selected['heading']:03d}°</span>",
            unsafe_allow_html=True)

        # Camera
        if selected["live_cam"]:
            cam_trigger = False
            if auto_capture_cam:
                if "auto_cam_last" not in st.session_state:
                    st.session_state.auto_cam_last = 0
                if time.time() - st.session_state.auto_cam_last > 3:
                    cam_trigger = True
                    st.session_state.auto_cam_last = time.time()

            cam_status = ":red_circle: REC" if auto_capture_cam else ":eye: IDLE"
            st.markdown(
                f"<div style='font-family:JetBrains Mono,monospace;color:#00D9FF;"
                f"font-size:0.82rem;margin-top:8px'>:movie_camera:  EYE  //  "
                f"NC-Conv-SSM  &nbsp;<span style='color:#5B7FA3'>{cam_status}</span></div>",
                unsafe_allow_html=True)
            if (cam_trigger or
                st.button("CAPTURE FRAME", key="eye_btn",
                          use_container_width=True)):
                try:
                    from drone_demo_vision import (load_ncconv_classifier,
                                                   classify, estimate_sigma)
                    if "vision_net" not in st.session_state:
                        net, status = load_ncconv_classifier()
                        st.session_state.vision_net = net
                    with st.spinner("프레임 캡처 + NC-Conv 분류 ..."):
                        pair = capture_camera(cam_idx)
                    if pair is None:
                        st.error(f"CAM {cam_idx} OFFLINE")
                    else:
                        rgb, frame = pair
                        label, conf = classify(
                            st.session_state.vision_net, frame)
                        sigma = estimate_sigma(frame)
                        st.image(rgb, use_container_width=True,
                                 caption=f"NC-Conv: {label} "
                                         f"({conf*100:.0f}%) | σ {sigma:.2f}")
                        st.session_state.log.append(
                            f"[{time.strftime('%H:%M:%S')}] "
                            f"{selected['id']} EYE: {label}")
                except Exception as e:
                    st.error(f"VISION FAULT: {e}")
        else:
            st.markdown(
                "<div style='font-family:JetBrains Mono,monospace;color:#5B7FA3;"
                "font-size:0.78rem;margin-top:8px'>:movie_camera:  EYE  //  "
                "<span style='color:#FFB400'>SIMULATED</span></div>",
                unsafe_allow_html=True)

        # Mic
        if selected["live_mic"]:
            mic_trigger = False
            if auto_capture_mic:
                if "auto_mic_last" not in st.session_state:
                    st.session_state.auto_mic_last = 0
                if time.time() - st.session_state.auto_mic_last > 3:
                    mic_trigger = True
                    st.session_state.auto_mic_last = time.time()

            mic_status = ":red_circle: LIVE" if auto_capture_mic else ":ear: IDLE"
            st.markdown(
                f"<div style='font-family:JetBrains Mono,monospace;color:#00D9FF;"
                f"font-size:0.82rem;margin-top:12px'>:microphone:  EAR  //  "
                f"NC-{kws_model.upper()}  &nbsp;<span style='color:#5B7FA3'>{mic_status}</span></div>",
                unsafe_allow_html=True)
            manual_btn = st.button("CAPTURE 1.0s AUDIO", key="ear_btn",
                                    use_container_width=True)
            if mic_trigger or manual_btn:
                # Auto mode = 0.5 s (UI 멈춤 최소화), 수동 = 1.0 s
                cap_sec = 0.5 if mic_trigger else 1.0
                try:
                    from drone_demo_kws import RealKWSClassifier
                    if ("kws_obj" not in st.session_state or
                        st.session_state.get("kws_tag") !=
                            (kws_model, kws_backbone)):
                        st.session_state.kws_obj = RealKWSClassifier(
                            model=kws_model, backbone=kws_backbone)
                        st.session_state.kws_tag = (kws_model,
                                                     kws_backbone)
                    with st.spinner(f"녹음 {cap_sec}s ..."):
                        rec = capture_mic(cap_sec, int(mic_channels),
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
                        f"<div style='padding:14px;border-radius:6px;"
                        f"background:linear-gradient(135deg,{col}33,{col}11);"
                        f"border:1px solid {col};color:#FFFFFF;text-align:center;"
                        f"font-family:Orbitron,monospace;"
                        f"font-size:1.4em;font-weight:700;letter-spacing:0.06em'>"
                        f"{kw_kr(label).upper()} <small style='font-size:0.6em;color:#5B7FA3'>"
                        f"CONF {conf:.2f}</small></div>",
                        unsafe_allow_html=True)
                    st.session_state.log.append(
                        f"[{time.strftime('%H:%M:%S')}] "
                        f"{selected['id']} EAR: {kw_kr(label)}")
                except Exception as e:
                    st.error(f"AUDIO FAULT: {e}")
        else:
            st.markdown(
                "<div style='font-family:JetBrains Mono,monospace;color:#5B7FA3;"
                "font-size:0.78rem;margin-top:12px'>:microphone:  EAR  //  "
                "<span style='color:#FFB400'>SIMULATED</span></div>",
                unsafe_allow_html=True)
            cur = sel_lab[min(scan_idx, len(sel_lab) - 1)]
            if cur:
                chips = []
                for _, _, kw in cur:
                    col = KW_COLOURS.get(kw, "#888")
                    chips.append(
                        "<span style='background:" + col +
                        "33;border:1px solid " + col +
                        ";color:" + col +
                        ";padding:3px 10px;border-radius:10px;"
                        "font-family:JetBrains Mono,monospace;"
                        "font-size:0.78em;font-weight:600;margin:2px 3px'>" +
                        kw_kr(kw) + "</span>")
                st.markdown(
                    "<div style='margin-top:6px'>" +
                    " ".join(chips) + "</div>",
                    unsafe_allow_html=True)

    # ---- Threat board ----
    st.markdown("<div class='sec-title'>:warning:  THREAT BOARD  // "
                f"{selected['id']}</div>",
                unsafe_allow_html=True)
    cur_tracks = sel_lab[min(scan_idx, len(sel_lab) - 1)]
    if not cur_tracks:
        st.markdown(
            "<div style='color:#5B7FA3;font-family:JetBrains Mono,monospace;"
            "font-size:0.85rem;text-align:center;padding:14px'>"
            "NO ACTIVE CONTACTS</div>",
            unsafe_allow_html=True)
    else:
        sorted_tr = sorted(cur_tracks,
                            key=lambda t: -threat_level(t[2]))
        for tid, deg, kw in sorted_tr:
            lvl = threat_level(kw)
            kw_label = kw_kr(kw)
            st.markdown(
                f"<div class='threat-row lvl{min(lvl,3)}'>"
                f"<span class='threat-lvl l{lvl}'>L{lvl}</span>"
                f"<b>TRK#{tid:02d}</b> &nbsp;|&nbsp; "
                f"BRG <b>{deg:+06.1f}°</b> &nbsp;|&nbsp; "
                f"CLASS <b>{kw_label}</b></div>",
                unsafe_allow_html=True)

    # ---- Polar (top, smaller) ----
    st.markdown("<div class='sec-title'>:bar_chart:  SPATIAL SPECTRUM</div>",
                unsafe_allow_html=True)
    pol_l, pol_c, pol_r = st.columns([1, 2, 1])
    with pol_c:
        st.pyplot(fig_polar(sel_hist, sel_lab, scan_idx,
                             selected["color"]),
                  clear_figure=True)

    # ---- Tracking timeline (full width, bottom hero) ----
    st.markdown(
        "<div class='sec-title'>:dart:  TRACKING TIMELINE  "
        "<span style='color:#5B7FA3;font-weight:400;font-size:0.7em'>"
        "  ―  표적 추적 + 음성 인식 시간선 (전체 임무)</span></div>",
        unsafe_allow_html=True)
    st.pyplot(fig_timeline(sel_hist, sel_lab, sel_sc, scan_idx,
                            selected["color"], show_gt, show_occl),
              clear_figure=True)

    # ---- Threat history sparkline ----
    if len(st.session_state.threat_history) > 2:
        sp_col1, sp_col2 = st.columns([1.0, 3.0])
        with sp_col1:
            st.markdown(
                "<div class='unit-line'><span class='k'>THREAT TREND (60s)</span></div>",
                unsafe_allow_html=True)
        with sp_col2:
            st.pyplot(fig_sparkline(
                st.session_state.threat_history,
                color="#FF3B5C" if total_threats >= 15
                else "#FFB400" if total_threats >= 8 else "#2EE6A6"),
                clear_figure=True)

    # ---- Comm log ----
    st.markdown("<div class='sec-title'>:scroll:  COMM // EVENT LOG</div>",
                unsafe_allow_html=True)
    if not st.session_state.log:
        st.caption("(no events)")
    else:
        log_text = "\n".join(reversed(st.session_state.log[-15:]))
        st.code(log_text, language="text")

    # ---- Footer ----
    st.markdown(
        "<div style='margin-top:20px;padding-top:14px;"
        "border-top:1px solid rgba(0,217,255,0.18);"
        "font-family:JetBrains Mono,monospace;font-size:0.7rem;"
        "color:#5B7FA3;text-align:center;letter-spacing:0.12em'>"
        "IRONDOME LATTICE  //  github.com/DrJinHoChoi/IronDome-DOA-Tracking  //  "
        "DUAL LICENSE (ACADEMIC + COMMERCIAL)  //  jinhochoi@smartear.co.kr"
        "</div>",
        unsafe_allow_html=True)


if __name__ == "__main__":
    main()
