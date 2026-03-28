#!/usr/bin/env python3
"""
SmartEar COP-DOA CEO MVP Demo
==============================
Interactive Streamlit demo showcasing COP-based DOA estimation.
Core pitch: M=4 cheap mics outperform M=8 conventional systems.

Run: streamlit run mvp_demo.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time

from iron_dome_sim.signal_model.array import UniformLinearArray
from iron_dome_sim.signal_model.signal_generator import generate_snapshots
from iron_dome_sim.doa import SubspaceCOP, MUSIC, Capon, COP_CBF, COP_MVDR
from iron_dome_sim.eval.metrics import rmse_doa, detection_rate

# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="SmartEar COP-DOA",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── ElevenLabs-inspired Warm Minimalism CSS ──
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* === Page Background === */
    .stApp {
        background-color: #FDFCFC;
    }

    /* Warm peach edge accents */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0; left: 0; bottom: 0;
        width: 4px;
        background: linear-gradient(180deg, #BEDBFF, #FDFCFC 80%);
        z-index: 999;
    }
    .stApp::after {
        content: '';
        position: fixed;
        top: 0; right: 0; bottom: 0;
        width: 4px;
        background: linear-gradient(180deg, #BEDBFF, #FDFCFC 80%);
        z-index: 999;
    }

    /* === Typography === */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    .main-title {
        text-align: center;
        font-size: 2.8em;
        font-weight: 300;
        color: #0C0A09;
        margin-bottom: 0;
        letter-spacing: -0.5px;
        line-height: 1.1;
    }
    .main-title span {
        color: #0447FF;
    }
    .sub-title {
        text-align: center;
        color: #777169;
        font-size: 1.15em;
        margin-top: 8px;
        font-weight: 400;
        letter-spacing: 0.5px;
    }

    /* === Cards === */
    .metric-card {
        background: #F5F3F1;
        border-radius: 20px;
        padding: 28px 16px;
        text-align: center;
        border: 1px solid #EBE8E4;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        border-color: #D4D0CC;
        box-shadow: 0 4px 24px rgba(0,0,0,0.04);
        transform: translateY(-2px);
    }
    .metric-value {
        font-size: 2.8em;
        font-weight: 700;
        color: #0C0A09;
        line-height: 1.1;
    }
    .metric-label {
        color: #777169;
        font-size: 0.85em;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 4px;
        font-weight: 500;
    }
    .metric-sub {
        color: #A59F97;
        font-size: 0.8em;
        margin-top: 4px;
    }

    .highlight-box {
        background: #F5F3F1;
        border-left: 3px solid #0447FF;
        padding: 18px 20px;
        border-radius: 0 16px 16px 0;
        margin: 12px 0;
        color: #44403B;
        border-top: 1px solid #EBE8E4;
        border-bottom: 1px solid #EBE8E4;
    }
    .highlight-box h4 {
        color: #0C0A09 !important;
        margin: 0 0 8px 0;
        font-weight: 600;
    }
    .highlight-box b {
        color: #0C0A09;
    }

    /* === Tabs === */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #F5F3F1;
        border-radius: 16px;
        padding: 4px;
        border: 1px solid #EBE8E4;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 500;
        letter-spacing: 0.3px;
        color: #57534E;
        border-radius: 12px;
        padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: #FFFFFF !important;
        color: #0C0A09 !important;
        border: 1px solid #EBE8E4;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }

    /* === Buttons === */
    .stButton > button {
        font-weight: 500;
        letter-spacing: 0.3px;
        border: 1px solid #EBE8E4;
        background: #FFFFFF;
        color: #0C0A09;
        border-radius: 9999px;
        padding: 8px 20px;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background: #0C0A09;
        color: #FFFFFF;
        border-color: #0C0A09;
    }

    /* === Slider & Inputs === */
    .stSlider > div > div > div > div {
        background: #0447FF !important;
    }
    .stSelectbox label, .stSlider label, .stRadio label {
        color: #44403B !important;
        font-weight: 500;
        letter-spacing: 0.3px;
    }

    /* === Metrics === */
    [data-testid="stMetricValue"] {
        font-weight: 700;
        color: #0C0A09;
    }
    [data-testid="stMetricLabel"] {
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #777169;
        font-weight: 500;
    }

    /* === Section Headers === */
    .stMarkdown h3 {
        color: #0C0A09 !important;
        letter-spacing: -0.3px;
        font-weight: 600;
        border-bottom: 1px solid #EBE8E4;
        padding-bottom: 10px;
    }

    /* === Divider === */
    hr {
        border-color: #EBE8E4 !important;
    }

    /* === Use-case cards === */
    .use-card {
        background: #FFFFFF;
        border: 1px solid #EBE8E4;
        border-radius: 20px;
        padding: 28px 16px;
        text-align: center;
        transition: all 0.3s ease;
    }
    .use-card:hover {
        transform: translateY(-3px);
        border-color: #D4D0CC;
        box-shadow: 0 8px 32px rgba(0,0,0,0.06);
    }
    .use-icon { font-size: 2.5em; margin-bottom: 8px; }
    .use-title {
        color: #0C0A09;
        font-weight: 700;
        font-size: 0.85em;
        letter-spacing: 1px;
        margin: 8px 0;
        text-transform: uppercase;
    }
    .use-desc {
        color: #777169;
        font-size: 0.9em;
        line-height: 1.5;
    }

    /* Accent dot */
    .dot-orange { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #DC2626; margin-right: 6px; }
    .dot-blue { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #0447FF; margin-right: 6px; }

    /* Footer */
    .footer-text {
        text-align: center;
        color: #A59F97;
        font-size: 0.85em;
        letter-spacing: 0.5px;
    }

    /* Alert boxes */
    .stAlert {
        border-radius: 12px;
    }

    /* Tag pill */
    .tag-cop {
        display: inline-block;
        background: #0447FF;
        color: #FFFFFF;
        padding: 2px 10px;
        border-radius: 9999px;
        font-size: 0.75em;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .tag-conv {
        display: inline-block;
        background: #EBE8E4;
        color: #57534E;
        padding: 2px 10px;
        border-radius: 9999px;
        font-size: 0.75em;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Helper Functions
# ============================================================
@st.cache_data
def run_doa_comparison(M, K, snr_db, T, seed=42):
    """Run DOA estimation with multiple algorithms."""
    np.random.seed(seed)
    array = UniformLinearArray(M=M, d=0.5)
    true_doas = np.radians(np.linspace(-60, 60, K))
    scan_angles = np.linspace(-np.pi / 2, np.pi / 2, 1801)

    X, _, _ = generate_snapshots(array, true_doas, snr_db, T, "non_stationary")

    results = {}
    alg_configs = {
        'MUSIC': lambda: MUSIC(array, num_sources=min(K, M - 1)),
        'Capon': lambda: Capon(array, num_sources=min(K, M - 1)),
        'COP-CBF': lambda: COP_CBF(array, num_sources=K, rho=2),
        'COP-MVDR': lambda: COP_MVDR(array, num_sources=K, rho=2),
        'COP-4th': lambda: SubspaceCOP(array, rho=2, num_sources=K, spectrum_type="combined"),
    }

    for name, make_alg in alg_configs.items():
        try:
            alg = make_alg()
            doa_est, spectrum = alg.estimate(X, scan_angles)
            rmse_val, _ = rmse_doa(doa_est, true_doas)
            pd, _ = detection_rate(doa_est, true_doas)
            results[name] = {
                'doas': np.degrees(doa_est),
                'spectrum': spectrum,
                'pd': pd,
                'rmse': np.degrees(rmse_val),
                'n_detected': len(doa_est),
            }
        except Exception as e:
            results[name] = {
                'doas': np.array([]),
                'spectrum': np.zeros(len(scan_angles)),
                'pd': 0.0,
                'rmse': 90.0,
                'n_detected': 0,
            }

    return results, np.degrees(true_doas), np.degrees(scan_angles)


@st.cache_data
def run_k_sweep(M, snr_db, T, K_range):
    """Sweep K for performance comparison."""
    all_results = {}
    for K in K_range:
        all_results[K] = run_doa_comparison(M, K, snr_db, T, seed=42 + K)
    return all_results


# ElevenLabs-style Plotly layout
CLEAN_LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor='#FFFFFF',
    plot_bgcolor='#FDFCFC',
    font=dict(family='Inter, -apple-system, sans-serif', color='#44403B', size=13),
    title_font=dict(family='Inter, sans-serif', color='#0C0A09', size=16, weight=600),
    xaxis=dict(gridcolor='#F5F3F1', zerolinecolor='#EBE8E4', linecolor='#EBE8E4'),
    yaxis=dict(gridcolor='#F5F3F1', zerolinecolor='#EBE8E4', linecolor='#EBE8E4'),
)

# Color palette: orange for COP (proposed), stone/gray for conventional
COLORS = {
    'MUSIC': '#A59F97',
    'Capon': '#57534E',
    'COP-CBF': '#3B82F6',
    'COP-MVDR': '#0447FF',
    'COP-4th': '#0C0A09',
}


def make_spectrum_plot(results, true_doas, scan_deg, title=""):
    """Create spectrum comparison — clean style."""
    fig = go.Figure()

    for name, data in results.items():
        spec = data['spectrum']
        if len(spec) > 0:
            spec_db = 10 * np.log10(np.abs(spec) / np.max(np.abs(spec) + 1e-30) + 1e-30)
            spec_db = np.clip(spec_db, -40, 0)
            is_cop = 'COP' in name
            fig.add_trace(go.Scatter(
                x=scan_deg, y=spec_db,
                name=f"{name} ({data['n_detected']}/{len(true_doas)})",
                line=dict(color=COLORS.get(name, '#A59F97'),
                          width=2.5 if is_cop else 1.2,
                          dash='solid' if is_cop else 'dot'),
                fill='tozeroy' if name == 'COP-4th' else None,
                fillcolor='rgba(12, 10, 9, 0.04)' if name == 'COP-4th' else None,
            ))

    for doa in true_doas:
        fig.add_vline(x=doa, line_dash="dash", line_color="#DC2626",
                      line_width=1, opacity=0.35)

    fig.update_layout(
        **CLEAN_LAYOUT,
        title=dict(text=title),
        xaxis_title="Angle (deg)",
        yaxis_title="Spectrum (dB)",
        height=450,
        legend=dict(font=dict(size=12), bgcolor='rgba(255,255,255,0.9)',
                    bordercolor='#EBE8E4', borderwidth=1),
        margin=dict(l=60, r=20, t=60, b=50),
    )
    return fig


def make_radar_plot(results, true_doas, title=""):
    """Create radar display — clean style."""
    fig = go.Figure()

    # Range rings
    for r in [0.25, 0.5, 0.75, 1.0]:
        theta_ring = np.linspace(0, 180, 90)
        fig.add_trace(go.Scatterpolar(
            r=[r] * 90, theta=theta_ring,
            mode='lines', line=dict(color='#EBE8E4', width=1),
            showlegend=False, hoverinfo='skip',
        ))

    # True DOAs
    for i_doa, doa in enumerate(true_doas):
        fig.add_trace(go.Scatterpolar(
            r=[1], theta=[doa + 90],
            mode='markers',
            marker=dict(size=14, color='#DC2626', symbol='star',
                       line=dict(color='rgba(220,38,38,0.3)', width=3)),
            name='True Target' if (i_doa == 0) else None,
            showlegend=bool(i_doa == 0),
        ))

    # Algorithm detections
    alg_styles = {
        'MUSIC': ('#A59F97', 'square', 0.85),
        'COP-MVDR': ('#0447FF', 'diamond', 0.7),
        'COP-4th': ('#0C0A09', 'circle', 0.55),
    }

    for name in ['MUSIC', 'COP-MVDR', 'COP-4th']:
        if name in results:
            data = results[name]
            color, symbol, r_val = alg_styles[name]
            for doa in data['doas']:
                fig.add_trace(go.Scatterpolar(
                    r=[r_val], theta=[doa + 90],
                    mode='markers',
                    marker=dict(size=11, color=color, symbol=symbol,
                               line=dict(color=color, width=1.5)),
                    name=f"{name}: {data['n_detected']}/{len(true_doas)}",
                    showlegend=False,
                ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color='#0C0A09')),
        polar=dict(
            bgcolor='#FDFCFC',
            radialaxis=dict(visible=False, range=[0, 1.15]),
            angularaxis=dict(
                tickvals=[0, 30, 60, 90, 120, 150, 180],
                ticktext=['-90°', '-60°', '-30°', '0°', '+30°', '+60°', '+90°'],
                gridcolor='#EBE8E4',
                linecolor='#D4D0CC',
                tickfont=dict(size=11, color='#777169'),
            ),
        ),
        paper_bgcolor='#FFFFFF',
        template="plotly_white",
        height=420,
        showlegend=True,
        legend=dict(font=dict(size=11), bgcolor='rgba(255,255,255,0.9)',
                    bordercolor='#EBE8E4', borderwidth=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def make_performance_bars(results, true_doas):
    """Create Pd/RMSE bar comparison — clean style."""
    names = list(results.keys())
    pds = [results[n]['pd'] for n in names]
    rmses = [results[n]['rmse'] for n in names]

    bar_colors = [COLORS.get(n, '#A59F97') for n in names]

    fig = make_subplots(rows=1, cols=2, subplot_titles=["Detection Rate (Pd)", "RMSE (deg)"])

    fig.add_trace(go.Bar(
        x=names, y=pds, marker_color=bar_colors,
        marker_line=dict(color='#EBE8E4', width=0),
        text=[f"{p:.0%}" for p in pds], textposition='auto',
        textfont=dict(size=13, color='white', weight=700),
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=names, y=rmses, marker_color=bar_colors,
        marker_line=dict(color='#EBE8E4', width=0),
        text=[f"{r:.1f}°" for r in rmses], textposition='auto',
        textfont=dict(size=13, color='white', weight=700),
    ), row=1, col=2)

    fig.update_layout(
        **CLEAN_LAYOUT,
        height=350,
        showlegend=False,
        margin=dict(l=40, r=20, t=40, b=50),
    )
    fig.update_xaxes(tickangle=30, tickfont=dict(size=11))
    return fig


@st.cache_data
def run_tracking_simulation(M, n_scans, snr_db, T, scenario_type="crossing"):
    """Run multi-target tracking simulation."""
    np.random.seed(42)
    array = UniformLinearArray(M=M, d=0.5)
    rho = 2
    scan_angles = np.linspace(-np.pi / 2, np.pi / 2, 361)
    dt = 0.1

    if scenario_type == "crossing":
        def get_doas(scan):
            d = []
            d.append(-40 + 70 * scan / n_scans)
            d.append(30 - 60 * scan / n_scans)
            if scan > n_scans * 0.4:
                d.append(50 - 10 * (scan - n_scans * 0.4) / (n_scans * 0.6))
            return np.array(d)
    elif scenario_type == "birth_death":
        def get_doas(scan):
            d = []
            d.append(-20 + 5 * np.sin(2 * np.pi * scan / n_scans))
            if scan < n_scans * 0.6:
                d.append(30 - 20 * scan / n_scans)
            if scan > n_scans * 0.3:
                d.append(-50 + 30 * (scan - n_scans * 0.3) / (n_scans * 0.7))
            if n_scans * 0.2 < scan < n_scans * 0.5:
                d.append(60)
            return np.array(d)
    else:
        def get_doas(scan):
            center = -30 + 60 * scan / n_scans
            d = [center + 8 * (i - 2) + 2 * np.sin(2 * np.pi * scan / (n_scans / 3) + i)
                 for i in range(5)]
            return np.array(d)

    tracks = []
    next_id = 0
    gate = 10.0

    history = {'scan': [], 'true_doas': [], 'cop_doas': [], 'tracks': [], 'spectrum': []}

    for scan in range(n_scans):
        true_doas_deg = get_doas(scan)
        true_doas_rad = np.radians(true_doas_deg)
        K = len(true_doas_deg)

        np.random.seed(42 + scan * 7)
        X, _, _ = generate_snapshots(array, true_doas_rad, snr_db, T, "non_stationary")

        try:
            cop = SubspaceCOP(array, rho=rho, num_sources=min(K, rho * (M - 1)),
                              spectrum_type="combined")
            doa_est, spectrum = cop.estimate(X, scan_angles)
            doas_deg = np.degrees(doa_est)
        except Exception:
            doas_deg = np.array([])
            spectrum = np.zeros(len(scan_angles))

        for tr in tracks:
            tr['theta'] += tr['vel'] * dt

        used_meas = set()
        for tr in tracks:
            if len(doas_deg) == 0:
                break
            dists = np.abs(doas_deg - tr['theta'])
            best_i = np.argmin(dists)
            if dists[best_i] < gate and best_i not in used_meas:
                innovation = doas_deg[best_i] - tr['theta']
                tr['vel'] = 0.7 * tr['vel'] + 0.3 * (innovation / max(dt, 0.01))
                tr['theta'] += 0.6 * innovation
                tr['weight'] = min(tr['weight'] + 0.15, 1.0)
                tr['hits'] += 1
                tr['miss'] = 0
                used_meas.add(best_i)
            else:
                tr['miss'] += 1
                tr['weight'] *= 0.8

        for i, d in enumerate(doas_deg):
            if i not in used_meas:
                tracks.append({
                    'id': next_id, 'theta': d, 'vel': 0.0,
                    'weight': 0.3, 'hits': 1, 'miss': 0
                })
                next_id += 1

        tracks = [tr for tr in tracks if tr['weight'] > 0.05 and tr['miss'] < 5]

        track_tuples = [(tr['theta'], tr['vel'], tr['weight'], tr['id']) for tr in tracks]
        history['scan'].append(scan)
        history['true_doas'].append(true_doas_deg.tolist())
        history['cop_doas'].append(doas_deg.tolist() if hasattr(doas_deg, 'tolist') else [])
        history['tracks'].append(track_tuples)
        history['spectrum'].append(spectrum)

    return history


def make_tracking_plot(history, current_scan=None):
    """Create tracking visualization — clean style."""
    n_scans = len(history['scan'])
    if current_scan is None:
        current_scan = n_scans - 1

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["DOA Track History", "Current Spectrum",
                        "Track Velocity", "Active Tracks"],
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "table"}]],
        vertical_spacing=0.12, horizontal_spacing=0.08,
    )

    # True DOAs
    for scan_i in range(current_scan + 1):
        for doa in history['true_doas'][scan_i]:
            fig.add_trace(go.Scatter(
                x=[scan_i], y=[doa],
                mode='markers', marker=dict(size=4, color='#0447FF', opacity=0.3),
                showlegend=False,
            ), row=1, col=1)

    # Tracks
    track_colors = ['#0447FF', '#DC2626', '#10B981', '#F59E0B', '#8B5CF6',
                    '#EC4899', '#2E86AB', '#0C0A09']
    track_histories = {}

    for scan_i in range(current_scan + 1):
        for theta, theta_dot, weight, label in history['tracks'][scan_i]:
            if weight >= 0.3:
                if label not in track_histories:
                    track_histories[label] = {'scans': [], 'doas': [], 'vels': []}
                track_histories[label]['scans'].append(scan_i)
                track_histories[label]['doas'].append(theta)
                track_histories[label]['vels'].append(theta_dot)

    for label, th in track_histories.items():
        color = track_colors[label % len(track_colors)]
        fig.add_trace(go.Scatter(
            x=th['scans'], y=th['doas'],
            mode='lines+markers',
            line=dict(color=color, width=2),
            marker=dict(size=4, color=color),
            name=f'Track {label}',
            showlegend=True,
        ), row=1, col=1)

    if current_scan < len(history['true_doas']):
        for doa in history['true_doas'][current_scan]:
            fig.add_hline(y=doa, line_dash="dot", line_color="#DC2626",
                         opacity=0.3, row=1, col=1)

    # Spectrum
    if current_scan < len(history['spectrum']):
        spec = history['spectrum'][current_scan]
        if len(spec) > 0:
            angles = np.linspace(-90, 90, len(spec))
            spec_db = 10 * np.log10(np.abs(spec) / (np.max(np.abs(spec)) + 1e-30) + 1e-30)
            spec_db = np.clip(spec_db, -40, 0)
            fig.add_trace(go.Scatter(
                x=angles, y=spec_db,
                mode='lines', line=dict(color='#0C0A09', width=2),
                name='COP Spectrum', showlegend=False,
            ), row=1, col=2)

            for doa in history['cop_doas'][current_scan]:
                fig.add_vline(x=doa, line_dash="solid", line_color="#0447FF",
                             line_width=1.5, opacity=0.6, row=1, col=2)

    # Velocity
    for label, th in track_histories.items():
        color = track_colors[label % len(track_colors)]
        fig.add_trace(go.Scatter(
            x=th['scans'], y=th['vels'],
            mode='lines', line=dict(color=color, width=1.5),
            name=f'T{label} vel', showlegend=False,
        ), row=2, col=1)

    # Table
    current_tracks = history['tracks'][current_scan] if current_scan < len(history['tracks']) else []
    confirmed = [(t, d, w, l) for t, d, w, l in current_tracks if w >= 0.5]

    if confirmed:
        fig.add_trace(go.Table(
            header=dict(
                values=['Track', 'DOA (deg)', 'Vel (deg/s)', 'Weight'],
                fill_color='#F5F3F1',
                font=dict(color='#0C0A09', size=12, weight=600),
                align='center',
                line_color='#EBE8E4',
            ),
            cells=dict(
                values=[
                    [f'T{l}' for _, _, _, l in confirmed],
                    [f'{t:.1f}' for t, _, _, _ in confirmed],
                    [f'{d:.2f}' for _, d, _, _ in confirmed],
                    [f'{w:.2f}' for _, _, w, _ in confirmed],
                ],
                fill_color='#FFFFFF',
                font=dict(color='#44403B', size=12),
                align='center',
                line_color='#EBE8E4',
            ),
        ), row=2, col=2)

    fig.update_layout(
        **CLEAN_LAYOUT,
        height=650,
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(font=dict(size=11), x=0.01, y=0.99,
                    bgcolor='rgba(255,255,255,0.9)', bordercolor='#EBE8E4', borderwidth=1),
    )
    fig.update_xaxes(title_text="Scan", row=1, col=1, gridcolor='#F5F3F1')
    fig.update_yaxes(title_text="DOA (deg)", row=1, col=1, gridcolor='#F5F3F1')
    fig.update_xaxes(title_text="Angle (deg)", row=1, col=2, gridcolor='#F5F3F1')
    fig.update_yaxes(title_text="dB", row=1, col=2, gridcolor='#F5F3F1')
    fig.update_xaxes(title_text="Scan", row=2, col=1, gridcolor='#F5F3F1')
    fig.update_yaxes(title_text="Velocity (deg/s)", row=2, col=1, gridcolor='#F5F3F1')

    return fig


# ============================================================
# Header
# ============================================================
st.markdown("""
<div style="text-align:center; padding: 40px 0 20px 0;">
    <p style="font-size:0.85em; font-weight:600; color:#0447FF; letter-spacing:3px; text-transform:uppercase; margin-bottom:12px;">SmartEar Technology</p>
    <p style="font-size:3.5em; font-weight:300; color:#0C0A09; margin:0; letter-spacing:-1.5px; line-height:1.05;">
        4 Mics. 6 Targets.<br><span style="color:#0447FF; font-weight:700;">Impossible</span> Made Real.
    </p>
    <p style="font-size:1.2em; color:#777169; margin-top:16px; font-weight:400;">
        COP-DOA breaks the physical limit &mdash; resolving more sources than sensors,<br>
        in real time, on a $4 chip.
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# Tabs
# ============================================================
tab1, tab2, tab_track, tab3, tab4 = st.tabs([
    "Live Radar", "Algorithm Showdown", "Multi-Target Tracking",
    "M=4 Small Array", "Benchmarks"
])

# ============================================================
# Tab 1: Live Radar
# ============================================================
with tab1:
    st.markdown("### Real-Time DOA Detection")

    col_ctrl, col_viz = st.columns([1, 3])

    with col_ctrl:
        st.markdown("**Scenario**")
        scenario = st.selectbox("Select", [
            "3 targets (Easy)",
            "5 targets (Hard — MUSIC fails)",
            "6 targets (Extreme — only COP)",
            "8 targets (COP-MVDR shines)",
        ], index=2)

        K_map = {
            "3 targets (Easy)": 3,
            "5 targets (Hard — MUSIC fails)": 5,
            "6 targets (Extreme — only COP)": 6,
            "8 targets (COP-MVDR shines)": 8,
        }
        K = K_map[scenario]

        M_radar = st.radio("Array size", [4, 8], index=0, horizontal=True)
        snr_radar = st.slider("SNR (dB)", -5, 20, 15)
        seed_radar = st.number_input("Random seed", 1, 999, 42)

    results, true_doas, scan_deg = run_doa_comparison(M_radar, K, snr_radar, 256, seed=seed_radar)

    with col_viz:
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            fig_radar = make_radar_plot(results, true_doas,
                                        f"Radar Display (M={M_radar}, K={K})")
            st.plotly_chart(fig_radar, use_container_width=True)
        with col_r2:
            fig_spec = make_spectrum_plot(results, true_doas, scan_deg,
                                          f"Spatial Spectrum (SNR={snr_radar}dB)")
            st.plotly_chart(fig_spec, use_container_width=True)

    # Scorecard
    st.markdown("---")
    cols = st.columns(len(results))
    for i, (name, data) in enumerate(results.items()):
        with cols[i]:
            is_cop = 'COP' in name
            if data['pd'] >= 0.8:
                pd_color = "#0447FF"
            elif data['pd'] >= 0.5:
                pd_color = "#F59E0B"
            else:
                pd_color = "#DC2626"
            tag = '<span class="tag-cop">PROPOSED</span>' if is_cop else '<span class="tag-conv">CONV.</span>'
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-weight:600; color:#44403B; margin-bottom:4px">{name} {tag}</div>
                <div class="metric-value" style="color: {pd_color}">{data['pd']:.0%}</div>
                <div class="metric-label">Detection Rate</div>
                <div class="metric-sub">{data['n_detected']}/{len(true_doas)} detected &middot; RMSE {data['rmse']:.1f}&deg;</div>
            </div>
            """, unsafe_allow_html=True)


# ============================================================
# Tab 2: Algorithm Showdown
# ============================================================
with tab2:
    st.markdown("### Interactive Algorithm Comparison")

    col_s1, col_s2 = st.columns([1, 3])

    with col_s1:
        st.markdown("**Parameters**")
        M_show = st.select_slider("Sensors (M)", options=[4, 6, 8], value=4)
        K_show = st.slider("Sources (K)", 2, 10, 5)
        snr_show = st.slider("SNR (dB)", -10, 20, 15, key="snr_show")
        T_show = st.select_slider("Snapshots (T)", options=[32, 64, 128, 256, 512], value=256)

        st.markdown("---")
        st.markdown("**Quick Presets**")
        if st.button("Easy (K=3, SNR=20)", use_container_width=True):
            st.session_state['preset'] = (4, 3, 20, 256)
            st.rerun()
        if st.button("Hard (K=5, SNR=5)", use_container_width=True):
            st.session_state['preset'] = (4, 5, 5, 256)
            st.rerun()
        if st.button("Impossible for MUSIC", use_container_width=True):
            st.session_state['preset'] = (4, 6, 10, 256)
            st.rerun()

        conv_limit = M_show - 1
        cop_limit = 2 * (M_show - 1)
        st.markdown(f"""
        <div class="highlight-box">
            <h4>M={M_show} Array Limits</h4>
            <span class="dot-orange"></span>MUSIC/Capon: K &le; <b>{conv_limit}</b><br>
            <span class="dot-blue"></span>COP (&rho;=2): K &le; <b>{cop_limit}</b><br>
            Virtual array: M<sub>v</sub> = <b>{cop_limit + 1}</b>
        </div>
        """, unsafe_allow_html=True)

        if K_show > conv_limit:
            st.warning(f"K={K_show} > M-1={conv_limit}: MUSIC/Capon will fail")
        if K_show > cop_limit:
            st.error(f"K={K_show} > COP limit={cop_limit}: Need SD-COP")

    results2, true_doas2, scan_deg2 = run_doa_comparison(M_show, K_show, snr_show, T_show)

    with col_s2:
        fig_s = make_spectrum_plot(results2, true_doas2, scan_deg2,
                                   f"M={M_show}, K={K_show}, SNR={snr_show}dB, T={T_show}")
        st.plotly_chart(fig_s, use_container_width=True)

        fig_bars = make_performance_bars(results2, true_doas2)
        st.plotly_chart(fig_bars, use_container_width=True)


# ============================================================
# Tab: Multi-Target Tracking
# ============================================================
with tab_track:
    st.markdown("### COP + GM-PHD Multi-Target Tracking")
    st.markdown("""
    <div class="highlight-box">
        <h4>Track, Detect, Resolve</h4>
        COP DOA + GM-PHD tracker: automatic <b>birth/death</b>, <b>crossing targets</b>,
        and <b>velocity estimation</b> &mdash; all with M=4 mics.
    </div>
    """, unsafe_allow_html=True)

    col_tc, col_tv = st.columns([1, 3])

    with col_tc:
        track_scenario = st.selectbox("Scenario", [
            "crossing", "birth_death", "swarm"
        ], format_func=lambda x: {
            "crossing": "Crossing Targets",
            "birth_death": "Birth & Death",
            "swarm": "Swarm (5 targets)",
        }[x])

        M_track = st.radio("Array (M)", [4, 8], index=0, horizontal=True, key="m_track")
        snr_track = st.slider("SNR (dB)", 0, 20, 15, key="snr_track")
        n_scans = st.slider("Scans", 30, 100, 60, key="n_scans")

        st.markdown("---")
        st.markdown(f"""
        <div class="highlight-box">
            <b>Config</b><br>
            M={M_track}, M<sub>v</sub>={2*(M_track-1)+1}<br>
            COP limit: K={2*(M_track-1)}<br>
            GM-PHD tracker
        </div>
        """, unsafe_allow_html=True)

    history = run_tracking_simulation(M_track, n_scans, snr_track, 64, track_scenario)

    with col_tv:
        scan_slider = st.slider("Scan frame", 0, n_scans - 1, n_scans - 1, key="scan_frame")
        fig_track = make_tracking_plot(history, current_scan=scan_slider)
        st.plotly_chart(fig_track, use_container_width=True)

    # Summary stats
    st.markdown("---")
    total_true = sum(len(d) for d in history['true_doas'])
    total_detected = sum(len(d) for d in history['cop_doas'])
    unique_tracks = set()
    for tracks_list in history['tracks']:
        for _, _, w, l in tracks_list:
            if w >= 0.5:
                unique_tracks.add(l)

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        st.metric("Total Scans", n_scans)
    with col_s2:
        st.metric("Avg True Sources", f"{total_true / n_scans:.1f}")
    with col_s3:
        st.metric("Avg Detections", f"{total_detected / n_scans:.1f}")
    with col_s4:
        st.metric("Unique Tracks", len(unique_tracks))


# ============================================================
# Tab 3: M=4 Small Array
# ============================================================
with tab3:
    st.markdown("### The Hardware Advantage")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
        <div class="highlight-box">
            <h4>Virtual Aperture Expansion</h4>
            4th-order cumulants create a <b>virtual array</b> with more elements than physical sensors.
        </div>
        """, unsafe_allow_html=True)

        fig_va = go.Figure()
        phys_pos = np.arange(4) * 0.5
        fig_va.add_trace(go.Scatter(
            x=phys_pos, y=[1] * 4,
            mode='markers+text',
            marker=dict(size=20, color='#0C0A09', symbol='circle'),
            text=[f'M{i+1}' for i in range(4)],
            textposition='top center',
            textfont=dict(color='#44403B'),
            name='Physical (M=4)',
        ))
        virt_pos = np.arange(7) * 0.5
        fig_va.add_trace(go.Scatter(
            x=virt_pos, y=[0] * 7,
            mode='markers+text',
            marker=dict(size=18, color='#0447FF', symbol='diamond'),
            text=[f'V{i+1}' for i in range(7)],
            textposition='bottom center',
            textfont=dict(color='#44403B'),
            name='Virtual (M_v=7)',
        ))
        fig_va.update_layout(
            title="Physical to Virtual Array",
            template="plotly_white",
            paper_bgcolor='#FFFFFF',
            plot_bgcolor='#FDFCFC',
            height=250,
            yaxis=dict(range=[-0.5, 1.8], visible=False),
            xaxis_title="Position (wavelengths)",
            font=dict(family='Inter, sans-serif', color='#44403B'),
            margin=dict(l=40, r=20, t=50, b=40),
        )
        st.plotly_chart(fig_va, use_container_width=True)

    with col_b:
        st.markdown("""
        <div class="highlight-box">
            <h4>Cost & Performance</h4>
        </div>
        """, unsafe_allow_html=True)

        cost_data = {
            'Metric': ['Microphones', 'Max Sources (Conv.)', 'Max Sources (COP)',
                       'Hardware Cost', 'Power', 'Cortex-M7 Latency', 'PCB Size'],
            'M=4 Array': ['4', '3', '6', '$4', '~50mW', '~12ms', '15x15mm'],
            'M=8 Array': ['8', '7', '14', '$12', '~120mW', '~45ms', '30x30mm'],
            'Advantage': ['2x fewer', 'COP closes gap', '2x COP range', '3x cheaper',
                         '2.4x less', '3.7x faster', '4x smaller'],
        }

        for i in range(len(cost_data['Metric'])):
            cols3 = st.columns([2, 1.5, 1.5, 2])
            with cols3[0]:
                st.markdown(f"**{cost_data['Metric'][i]}**")
            with cols3[1]:
                st.markdown(f"`{cost_data['M=4 Array'][i]}`")
            with cols3[2]:
                st.markdown(f"`{cost_data['M=8 Array'][i]}`")
            with cols3[3]:
                st.markdown(f"{cost_data['Advantage'][i]}")

    st.markdown("---")

    st.markdown("### Target Applications")
    col_u1, col_u2, col_u3, col_u4 = st.columns(4)

    use_cases = [
        ("drone", "Drone Detection", "360 deg surveillance with 4 mics. Multiple drones simultaneously."),
        ("speaker", "Smart Speaker", "Multi-user voice tracking. 6 users with 4 mics."),
        ("hearing", "Hearing Aid", "Selective listening. Ultra-low power COP."),
        ("factory", "Industrial IoT", "Acoustic anomaly detection. Multi-source localization."),
    ]
    icons = {"drone": "&#9992;", "speaker": "&#128266;", "hearing": "&#129467;", "factory": "&#127981;"}
    for i, (key, title, desc) in enumerate(use_cases):
        with [col_u1, col_u2, col_u3, col_u4][i]:
            st.markdown(f"""
            <div class="use-card">
                <div style="font-size:2.5em; margin-bottom:8px">{icons[key]}</div>
                <div class="use-title">{title}</div>
                <div class="use-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### M=4 Live Comparison")
    K_m4 = st.slider("Number of sources", 2, 8, 5, key="k_m4")
    results_m4, true_m4, scan_m4 = run_doa_comparison(4, K_m4, 15, 256, seed=77)

    col_m4a, col_m4b = st.columns(2)
    with col_m4a:
        fig_m4s = make_spectrum_plot(results_m4, true_m4, scan_m4,
                                     f"M=4, K={K_m4}: Spectrum")
        st.plotly_chart(fig_m4s, use_container_width=True)
    with col_m4b:
        fig_m4b = make_performance_bars(results_m4, true_m4)
        st.plotly_chart(fig_m4b, use_container_width=True)


# ============================================================
# Tab 4: Benchmark Results
# ============================================================
with tab4:
    st.markdown("### Benchmark Results")

    bench_tab1, bench_tab2 = st.tabs(["M=8 Results", "M=4 Results"])

    with bench_tab1:
        fig_dir_m8 = os.path.join(os.path.dirname(__file__), 'results', 'figures')
        if os.path.exists(fig_dir_m8):
            m8_figs = [
                ('fig_summary_combined.png', 'COP Family Performance Summary (M=8)'),
                ('fig1_k_scaling_pd.png', 'K Scaling: Detection Rate'),
                ('fig4_snr_rmse.png', 'SNR Robustness: RMSE'),
                ('fig5_resolution.png', 'Close-Spacing Resolution'),
            ]
            for fname, caption in m8_figs:
                fpath = os.path.join(fig_dir_m8, fname)
                if os.path.exists(fpath):
                    st.image(fpath, caption=caption, use_container_width=True)
                    st.markdown("---")

    with bench_tab2:
        fig_dir_m4 = os.path.join(os.path.dirname(__file__), 'results', 'figures_m4')
        if os.path.exists(fig_dir_m4):
            m4_figs = [
                ('fig_m4_summary.png', 'COP Family Performance Summary (M=4)'),
                ('fig1_m4_k_scaling_pd.png', 'K Scaling: Detection Rate (M=4)'),
                ('fig3_m4_snr_pd.png', 'SNR Robustness (M=4)'),
                ('fig5_m4_resolution.png', 'Close-Spacing Resolution (M=4)'),
            ]
            for fname, caption in m4_figs:
                fpath = os.path.join(fig_dir_m4, fname)
                if os.path.exists(fpath):
                    st.image(fpath, caption=caption, use_container_width=True)
                    st.markdown("---")

# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.markdown("""
<div class="footer-text">
    SmartEar Co., Ltd. &nbsp;&middot;&nbsp; COP-DOA Technology Demo &nbsp;&middot;&nbsp; Confidential<br>
    <i>COP-RFS: Higher-Order Cumulant DOA Estimation with Multi-Target Tracking</i><br>
    IEEE Trans. Signal Processing (Under Review)
</div>
""", unsafe_allow_html=True)
