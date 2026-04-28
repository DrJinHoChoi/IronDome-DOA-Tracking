"""드론 함대 통합 운용 콘솔 (browser dashboard).

NC 모델 패밀리 통합:
    눈 (영상)   : NC-Conv-SSM
    귀 (음향)   : NC-SSM / NC-TCN  (키워드 인식)
    공간 인식    : Mamba-COP-RFS  (8채널 마이크 어레이 DOA + GM-PHD 추적)

실행:
    streamlit run drone_demo_streamlit.py
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


st.set_page_config(page_title="드론 함대 운용 콘솔",
                   page_icon=":helicopter:", layout="wide")


# CSS — 군용 콘솔 스타일
st.markdown("""
<style>
.main { background-color: #0A0E1A; }
section[data-testid="stSidebar"] { background-color: #0E1A2C; }
div[data-testid="metric-container"] {
    background-color: #14213D;
    border: 1px solid #1E3A5F;
    padding: 8px;
    border-radius: 4px;
}
.fleet-card {
    background: #14213D;
    border: 2px solid #1E3A5F;
    border-radius: 6px;
    padding: 10px;
    color: white;
    text-align: center;
}
.fleet-card.selected { border-color: #FFD60A; }
.threat-row {
    background: #1A1F2E;
    border-left: 4px solid #E74C3C;
    padding: 6px 10px;
    margin: 3px 0;
    color: white;
    border-radius: 3px;
    font-family: 'Consolas', monospace;
}
.alert-banner {
    background: linear-gradient(90deg, #C0392B 0%, #E74C3C 100%);
    color: white;
    padding: 10px;
    border-radius: 4px;
    font-weight: 700;
    font-size: 1.1em;
    text-align: center;
    margin-bottom: 10px;
}
.status-ok    { color: #2ECC71; font-weight: 700; }
.status-warn  { color: #F39C12; font-weight: 700; }
.status-alert { color: #E74C3C; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------- #
# 함대 정의                                                              #
# --------------------------------------------------------------------- #
def default_fleet():
    return [
        {"id": "알파-1", "code": "A1", "pos": (-30, 20), "heading": 30,
         "scenario": "sar", "live_cam": True,  "live_mic": True,
         "color": "#2ECC71", "battery": 87, "alt": 120, "speed": 12,
         "role": "운용자기"},
        {"id": "알파-2", "code": "A2", "pos": (15, 35),   "heading": 270,
         "scenario": "sar", "live_cam": False, "live_mic": False,
         "color": "#3498DB", "battery": 73, "alt": 95, "speed": 14,
         "role": "구조 정찰"},
        {"id": "브라보-1", "code": "B1", "pos": (40, -10),  "heading": 200,
         "scenario": "combat", "live_cam": False, "live_mic": False,
         "color": "#E67E22", "battery": 64, "alt": 200, "speed": 18,
         "role": "전투 정찰"},
        {"id": "브라보-2", "code": "B2", "pos": (-15, -25), "heading": 110,
         "scenario": "combat", "live_cam": False, "live_mic": False,
         "color": "#E74C3C", "battery": 41, "alt": 180, "speed": 20,
         "role": "전투 정찰"},
    ]


KW_KR = {
    "help": "구조요청", "water": "물/식수", "down": "아래",
    "here": "여기있음", "vehicle": "차량", "drone": "드론음",
    "shot": "총성", "yes": "예", "no": "아니오", "up": "위",
    "left": "좌", "right": "우", "on": "켜짐", "off": "꺼짐",
    "stop": "정지", "go": "출발", "silence": "정적",
    "unknown": "미분류",
}


def kw_kr(kw):
    return KW_KR.get(kw, kw or "?")


def threat_level(kw):
    """키워드별 위협도 (1=낮음 5=긴급)."""
    t = {"help": 5, "shot": 5, "vehicle": 4, "drone": 4,
         "down": 3, "water": 2, "here": 3}
    return t.get(kw, 1)


def threat_color(level):
    return ["#7F8C8D", "#3498DB", "#27AE60", "#F39C12",
            "#E67E22", "#E74C3C"][level]


# --------------------------------------------------------------------- #
# 시뮬레이션 캐시                                                         #
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


@st.cache_data(show_spinner="공간 인식 파이프라인 실행 중...")
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
def fig_tactical_map(fleet, selected_id, fleet_data, scan_idx):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.set_facecolor("#0A0E1A")
    fig.patch.set_facecolor("#0A0E1A")
    ax.set_xlim(-65, 65); ax.set_ylim(-55, 55)
    ax.set_aspect("equal")

    # 격자 / 거리원
    for r in (10, 20, 30, 40, 50):
        circ = plt.Circle((0, 0), r, fill=False, color="#1E3A5F",
                          lw=0.7, alpha=0.7, linestyle="--")
        ax.add_patch(circ)
        ax.text(r * 0.7, -r * 0.7, f"{r}km", color="#1E3A5F",
                fontsize=7, ha="center")
    ax.axhline(0, color="#1E3A5F", lw=0.5, alpha=0.5)
    ax.axvline(0, color="#1E3A5F", lw=0.5, alpha=0.5)
    # 방위 표시
    for ang, lab in [(0, "N"), (90, "E"), (180, "S"), (270, "W")]:
        x = 55 * np.sin(np.deg2rad(ang))
        y = 55 * np.cos(np.deg2rad(ang))
        ax.text(x, y, lab, color="#7FB3D5", fontsize=12,
                fontweight="bold", ha="center", va="center")

    # 표적 집계
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
            target_xy.append((tx, ty, kw, d["color"]))
            ax.plot([dx, tx], [dy, ty],
                    color=KW_COLOURS.get(kw, "#888"),
                    lw=1.0, alpha=0.40, linestyle=":")

    # 표적 마커
    for x, y, kw, dcol in target_xy:
        col = KW_COLOURS.get(kw, "#888")
        lvl = threat_level(kw)
        size = 80 + lvl * 30
        ax.scatter([x], [y], s=size, color=col, marker="X",
                   edgecolors="white", linewidths=1.5, zorder=5,
                   alpha=0.95)
        if lvl >= 4:
            ring = plt.Circle((x, y), 3, fill=False, color="#E74C3C",
                              lw=1.5, alpha=0.7)
            ax.add_patch(ring)

    # 드론 그리기
    for d in fleet:
        x, y = d["pos"]
        is_sel = (d["id"] == selected_id)
        theta = np.deg2rad(d["heading"])
        # FOV cone
        for sweep in np.linspace(-np.pi / 2, np.pi / 2, 12):
            ax.plot([x, x + 28 * np.cos(theta + sweep)],
                    [y, y + 28 * np.sin(theta + sweep)],
                    color=d["color"], lw=0.4, alpha=0.10)
        # 드론 마커 (삼각형)
        size = 4
        tri = np.array([[size, 0], [-size * 0.6, size * 0.6],
                         [-size * 0.6, -size * 0.6]])
        c, s = np.cos(theta), np.sin(theta)
        rot = np.array([[c, -s], [s, c]])
        tri = (rot @ tri.T).T + np.array([x, y])
        poly = plt.Polygon(tri, color=d["color"],
                            ec="#FFD60A" if is_sel else "white",
                            lw=2.5 if is_sel else 0.8, zorder=6)
        ax.add_patch(poly)
        ax.text(x, y - 7, d["code"], color=d["color"],
                fontsize=10, fontweight="bold", ha="center",
                bbox=dict(facecolor="#0A0E1A", edgecolor="none",
                          alpha=0.85, pad=2))
        if is_sel:
            ax.scatter([x], [y], s=500, facecolors="none",
                       edgecolors="#FFD60A", lw=2, zorder=7,
                       linestyle="--")

    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"전술 상황도  ―  스캔 {scan_idx:02d}  ―  "
                 f"드론 {len(fleet)}기 / 표적 {len(target_xy)}건",
                 color="#FFD60A", fontsize=12, fontweight="bold",
                 family="monospace")
    for spine in ax.spines.values():
        spine.set_color("#1E3A5F")
    fig.tight_layout()
    return fig


def fig_polar_for(history, labelled, scan_idx, drone_color):
    angles = history["scan_angles_deg"]
    P = history["spectra"][scan_idx]
    P = P / (P.max() + 1e-9)
    fig = plt.figure(figsize=(5.2, 4.4))
    fig.patch.set_facecolor("#0A0E1A")
    ax = fig.add_subplot(111, projection="polar")
    ax.set_facecolor("#14213D")
    ax.plot(np.deg2rad(angles), P, color=drone_color, lw=1.6)
    ax.fill(np.deg2rad(angles), P, color=drone_color, alpha=0.18)
    for tid, deg, kw in labelled[scan_idx]:
        col = KW_COLOURS.get(kw, "#888")
        ax.plot([np.deg2rad(deg)], [1.0], "o", color=col, ms=12,
                markeredgecolor="white", markeredgewidth=1.5)
        ax.text(np.deg2rad(deg), 1.22, kw_kr(kw), color=col,
                fontweight="bold", fontsize=10, ha="center")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetamin(-90); ax.set_thetamax(90)
    ax.set_yticklabels([])
    ax.tick_params(colors="white")
    ax.set_title("공간 스펙트럼 (8채널 어레이)",
                 color="white", fontsize=10, fontweight="bold", pad=10,
                 family="monospace")
    fig.tight_layout()
    return fig


def fig_timeline(history, labelled, scenario, scan_idx, color,
                 show_gt, show_occl):
    n = scenario["n_scans"]
    angles = history["scan_angles_deg"]
    fig, ax = plt.subplots(figsize=(13, 2.6))
    fig.patch.set_facecolor("#0A0E1A")
    ax.set_facecolor("#14213D")
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
                ax.text(s, d + 6, kw_kr(kw), fontsize=7, color=col,
                        fontweight="bold", ha="center")
                prev = kw
    if show_gt:
        gt = history["gt"]
        for k in range(gt.shape[1]):
            ax.plot(np.arange(n), gt[:, k], color="#7F8C8D",
                    lw=0.6, linestyle="--", alpha=0.4)
    if show_occl:
        for sidx, a, b in scenario["occlusions"]:
            ax.axvspan(a, b, alpha=0.20, color="#F39C12")
    ax.axvline(scan_idx, color=color, lw=2, alpha=0.85)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(angles[0], angles[-1])
    ax.set_xlabel("스캔 시간축", color="white")
    ax.set_ylabel("방위 (도)", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#1E3A5F")
    ax.grid(alpha=0.2, color="#1E3A5F")
    ax.set_title("표적 추적 + 음성 인식 시간선",
                 color="white", fontsize=10, fontweight="bold",
                 family="monospace")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------- #
# 라이브 캡처                                                            #
# --------------------------------------------------------------------- #
def capture_camera(idx):
    import cv2
    cam = cv2.VideoCapture(int(idx))
    ok, frame = cam.read()
    cam.release()
    if not ok or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), frame


def capture_mic(seconds, channels, device):
    import sounddevice as sd
    n = int(seconds * 16000)
    return sd.rec(n, samplerate=16000, channels=channels,
                  device=device, dtype="float32", blocking=True)


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
            f"[{time.strftime('%H:%M:%S')}] 시스템 가동 -- 함대 4기 정상"
        ]
    if "mission_start" not in st.session_state:
        st.session_state.mission_start = time.time()

    fleet = st.session_state.fleet

    # ---- 사이드바 (운용 설정) ----
    st.sidebar.markdown("## :satellite_antenna: 운용 설정")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**임무 환경**")
    snr_db = st.sidebar.slider("음향 SNR (dB)", 0.0, 20.0, 12.0, 0.5,
                                help="환경 잡음 대비 신호 강도")
    n_scans = st.sidebar.slider("임무 스캔 수", 10, 60, 30,
                                 help="시뮬레이션 길이")
    show_gt = st.sidebar.toggle("실측 위치 표시 (디버그용)", value=False)
    show_occl = st.sidebar.toggle("차폐 구간 음영", value=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**음성 인식 엔진 (귀)**")
    use_real = st.sidebar.toggle("실제 NC-SSM/NC-TCN 추론 사용",
                                  value=False,
                                  help="끄면 GT 룩업(즉시), 켜면 모델 추론")
    kws_model = st.sidebar.selectbox("모델", ["nc-ssm", "nc-tcn"],
                                     disabled=not use_real)
    kws_backbone = st.sidebar.selectbox("백본", ["Tiny", "Small"],
                                        disabled=not use_real or
                                        kws_model != "nc-ssm")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**라이브 하드웨어 (운용자기 전용)**")
    cam_idx = st.sidebar.number_input("카메라 인덱스", 0, 8, 0)
    mic_channels = st.sidebar.number_input("마이크 채널 수", 1, 8, 4)
    mic_device = st.sidebar.number_input("마이크 장치 인덱스", 0, 32, 1)

    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 임무 재실행", type="primary",
                          use_container_width=True):
        st.session_state.rerun_nonce += 1
        st.session_state.mission_start = time.time()
        st.session_state.log.append(
            f"[{time.strftime('%H:%M:%S')}] 임무 재시작")

    # ---- 함대 데이터 ----
    fleet_data = {}
    for d in fleet:
        fleet_data[d["id"]] = cached_run(
            d["scenario"], snr_db, n_scans,
            "real" if use_real else "stub", kws_model, kws_backbone,
            d["id"], st.session_state.rerun_nonce)

    # ---- 헤더: 임무 상태 ----
    elapsed = int(time.time() - st.session_state.mission_start)
    h, m, s = elapsed // 3600, (elapsed // 60) % 60, elapsed % 60

    head1, head2, head3, head4, head5, head6 = st.columns([1.6, 1, 1, 1, 1, 1])
    with head1:
        st.markdown("# :helicopter: 드론 함대 운용 콘솔")
        st.caption("Drone Fleet Operations Console — IronDome 시리즈")
    head2.metric("임무 시간", f"{h:02d}:{m:02d}:{s:02d}")
    head3.metric("함대 가동", f"{len(fleet)}기", "정상")
    total_threats = sum(
        sum(threat_level(kw) for _, _, kw in lab[min(scan_idx, len(lab) - 1)])
        for d in fleet
        for scan_idx in [st.session_state.get("scan_idx", n_scans // 2)]
        for sc, hist, lab in [fleet_data[d["id"]]]
    )
    head4.metric("위협 점수", f"{total_threats}",
                 "긴급" if total_threats >= 15 else
                 "주의" if total_threats >= 8 else "정상")
    head5.metric("AI 모델 풋프린트", "< 320 KB", "INT8")
    head6.metric("스캔 지연", "3.4 ms", "Cortex-M7")

    # ---- 경보 배너 (위협 점수 기반) ----
    if total_threats >= 15:
        st.markdown(
            "<div class='alert-banner'>:rotating_light: "
            "긴급 경보 — 다수 고위협 표적 탐지. 즉시 대응 권고.</div>",
            unsafe_allow_html=True)
    elif total_threats >= 8:
        st.markdown(
            "<div class='alert-banner' style='background:"
            "linear-gradient(90deg,#D68910 0%,#F39C12 100%);'>"
            ":warning: 주의 — 다수 표적 추적 중. 임무 진행 모니터링.</div>",
            unsafe_allow_html=True)

    st.markdown("---")

    # ---- 함대 카드 ----
    st.markdown("### :flag-kr: 함대 상태")
    cols = st.columns(len(fleet))
    for i, d in enumerate(fleet):
        sc, hist, lab = fleet_data[d["id"]]
        scan_idx = st.session_state.get("scan_idx", n_scans // 2)
        scan_idx = min(scan_idx, n_scans - 1)
        n_now = len(lab[scan_idx])
        is_sel = (d["id"] == st.session_state.selected)
        bat_status = ("status-ok" if d["battery"] > 60 else
                      "status-warn" if d["battery"] > 30 else
                      "status-alert")
        with cols[i]:
            border = "3px solid #FFD60A" if is_sel else \
                     f"2px solid {d['color']}"
            html = (
                f"<div style='background:#14213D;border:{border};"
                f"border-radius:8px;padding:12px;color:white;"
                f"font-family:monospace;'>"
                f"<div style='color:{d['color']};font-size:1.1em;"
                f"font-weight:700'>{d['id']} ({d['code']})</div>"
                f"<div style='color:#7FB3D5;font-size:0.85em;"
                f"margin-bottom:8px'>{d['role']}</div>"
                f"<hr style='border:0;border-top:1px solid #1E3A5F'>"
                f"<div>:battery: <span class='{bat_status}'>"
                f"{d['battery']}%</span></div>"
                f"<div>고도: {d['alt']} m</div>"
                f"<div>속도: {d['speed']} m/s</div>"
                f"<div>임무: {sc['name']}</div>"
                f"<div style='margin-top:6px;color:#FFD60A'>"
                f":dart: 현재 탐지 <b>{n_now}</b> / "
                f"누적 {sum(len(s) for s in lab)}</div>"
                f"</div>"
            )
            st.markdown(html, unsafe_allow_html=True)
            if st.button(("✅ 선택됨" if is_sel else "➤ 선택"),
                         key=f"sel_{d['id']}",
                         use_container_width=True,
                         type="secondary" if is_sel else "primary"):
                st.session_state.selected = d["id"]
                st.session_state.log.append(
                    f"[{time.strftime('%H:%M:%S')}] {d['id']} 선택됨")

    selected = next(d for d in fleet
                     if d["id"] == st.session_state.selected)
    sel_sc, sel_hist, sel_lab = fleet_data[selected["id"]]

    st.markdown("---")

    # ---- 메인: 전술도 + 운용자기 라이브 ----
    map_col, live_col = st.columns([1.4, 1.0])

    with map_col:
        st.markdown("### :compass: 전술 상황도")
        scan_idx = st.slider("임무 스캔 시점", 0, max(n_scans - 1, 0),
                              value=min(n_scans // 2, n_scans - 1),
                              key="scan_idx")
        st.pyplot(fig_tactical_map(fleet, selected["id"], fleet_data,
                                    scan_idx),
                  clear_figure=True)

    with live_col:
        st.markdown(f"### :round_pushpin: {selected['id']} 상세")
        st.caption(f"임무: **{sel_sc['name']}** | 위치: "
                   f"({selected['pos'][0]}, {selected['pos'][1]}) | "
                   f"방향: {selected['heading']}°")

        if selected["live_cam"]:
            st.markdown("**:eye: 카메라 (라이브)**")
            if st.button(":camera_with_flash: 영상 캡처 + 분류",
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
                        st.error(f"카메라 {cam_idx}번 응답 없음")
                    else:
                        rgb, frame = pair
                        label, conf = classify(
                            st.session_state.vision_net, frame)
                        sigma = estimate_sigma(frame)
                        st.image(rgb, use_container_width=True,
                                 caption=f"NC-Conv 분류: {label} "
                                         f"({conf*100:.0f}%)  |  "
                                         f"σ-게이트: {sigma:.2f}")
                        st.session_state.log.append(
                            f"[{time.strftime('%H:%M:%S')}] "
                            f"{selected['id']} 영상: {label}")
                except Exception as e:
                    st.error(f"영상 처리 오류: {e}")
        else:
            st.markdown("**:eye: 카메라 (시뮬레이션)**")
            st.info("이 드론은 가상 영상 사용 중\n\n"
                    "실배치 시: 드론 탑재 카메라로 NC-Conv 247KB 추론")

        if selected["live_mic"]:
            st.markdown("**:ear: 마이크 (라이브)**")
            if st.button(":microphone: 1.5초 음성 캡처 + 인식",
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
                        f"<div style='padding:14px;border-radius:8px;"
                        f"background:{col};color:white;text-align:center;"
                        f"font-size:1.4em;font-weight:700'>"
                        f"{kw_kr(label)} <small>(신뢰도 "
                        f"{conf:.2f})</small></div>",
                        unsafe_allow_html=True)
                    st.caption("채널별 RMS: " + " ".join(
                        f"ch{i}={r:.3f}" for i, r in enumerate(rms)))
                    st.session_state.log.append(
                        f"[{time.strftime('%H:%M:%S')}] "
                        f"{selected['id']} 음성: {kw_kr(label)}")
                except Exception as e:
                    st.error(f"마이크 오류: {e}")
        else:
            st.markdown("**:ear: 마이크 (시뮬레이션)**")
            cur = sel_lab[min(scan_idx, len(sel_lab) - 1)]
            if cur:
                chips = []
                for _, _, kw in cur:
                    col = KW_COLOURS.get(kw, "#888")
                    chips.append(
                        "<span style='background:" + col +
                        ";color:white;padding:3px 10px;"
                        "border-radius:10px;font-size:0.9em;"
                        "font-weight:600;margin:2px'>" +
                        kw_kr(kw) + "</span>")
                st.markdown(" ".join(chips), unsafe_allow_html=True)
            else:
                st.info("이 스캔에 탐지 없음")

        st.markdown("**드론 상태**")
        s1, s2, s3 = st.columns(3)
        s1.metric("배터리", f"{selected['battery']}%")
        s2.metric("고도", f"{selected['alt']} m")
        s3.metric("속도", f"{selected['speed']} m/s")

    st.markdown("---")

    # ---- 위협 보드 ----
    st.markdown("### :warning: 위협 보드 — 현재 탐지 (선택 드론)")
    cur_tracks = sel_lab[min(scan_idx, len(sel_lab) - 1)]
    if not cur_tracks:
        st.info("선택 드론에서 탐지된 표적 없음.")
    else:
        sorted_tr = sorted(cur_tracks,
                            key=lambda t: -threat_level(t[2]))
        for tid, deg, kw in sorted_tr:
            lvl = threat_level(kw)
            col = threat_color(lvl)
            kw_label = kw_kr(kw)
            st.markdown(
                f"<div class='threat-row' style='border-left-color:{col}'>"
                f"<b style='color:{col}'>위협도 {lvl}</b>  |  "
                f"트랙 #{tid:02d}  |  방위 <b>{deg:+6.1f}°</b>  |  "
                f"분류: <b>{kw_label}</b></div>",
                unsafe_allow_html=True)

    st.markdown("---")

    # ---- 폴라 + 타임라인 ----
    p_col, t_col = st.columns([1.0, 1.7])
    with p_col:
        st.pyplot(fig_polar_for(sel_hist, sel_lab, scan_idx,
                                 selected["color"]),
                  clear_figure=True)
    with t_col:
        st.pyplot(fig_timeline(sel_hist, sel_lab, sel_sc, scan_idx,
                                selected["color"], show_gt, show_occl),
                  clear_figure=True)

    # ---- 통신 / 이벤트 로그 ----
    st.markdown("---")
    st.markdown("### :scroll: 통신 / 이벤트 로그")
    if not st.session_state.log:
        st.caption("(이벤트 없음)")
    else:
        log_text = "\n".join(reversed(st.session_state.log[-15:]))
        st.code(log_text, language="text")

    # ---- 푸터 ----
    st.markdown("---")
    st.caption(
        "🚁 드론 함대 운용 콘솔 — IronDome-DOA-Tracking + NC family   |   "
        "github.com/DrJinHoChoi/IronDome-DOA-Tracking   |   "
        "이중 라이선스 (학술+상업)   |   "
        "문의: jinhochoi@smartear.co.kr")


if __name__ == "__main__":
    main()
