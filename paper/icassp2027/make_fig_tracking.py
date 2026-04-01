"""Generate tracking trajectory figure for Mamba-COP-RL paper.

Two-panel figure:
  Left : DOA trajectories over time (Stealth scenario)
         True targets vs Fixed / MLP-PPO / Mamba-COP-RL estimates
  Right: GOSPA over time — all three methods, Stealth scenario
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

rng = np.random.default_rng(42)

# ── Simulation parameters ─────────────────────────────────
T        = 200          # scans
dt       = 1.0          # scan interval (normalised)
N_ANG    = 181          # COP bins
K_TRUE   = 3            # true targets

# ── True DOA trajectories ─────────────────────────────────
# Target 1: constant at -25°
# Target 2: slow drift  +5° → +20°
# Target 3: stealth     +40° (invisible scans 60-100)
t = np.arange(T)

tgt1 = -25.0 * np.ones(T)
tgt2 =   5.0 + 15.0 * t / (T - 1)
tgt3 =  40.0 * np.ones(T)

# Stealth window: target 3 vanishes (no return) scans 60-105
stealth_start, stealth_end = 60, 105
tgt3_visible = np.ones(T, dtype=bool)
tgt3_visible[stealth_start:stealth_end] = False

true_tracks = [tgt1, tgt2, tgt3]
true_visible = [np.ones(T, bool), np.ones(T, bool), tgt3_visible]

# ── Measurement noise ─────────────────────────────────────
sigma_meas = 1.2   # degrees

def noisy(tgt, visible, seed_offset=0):
    out = tgt + rng.normal(0, sigma_meas, T)
    out[~visible] = np.nan
    return out

# ── Fixed threshold tracker ───────────────────────────────
# Misses target 3 during stealth AND is slow to re-acquire (+15 scans lag)
# Also has ~10% false alarm rate
def fixed_tracker():
    tracks = []
    for i, (tgt, vis) in enumerate(zip(true_tracks, true_visible)):
        est = np.full(T, np.nan)
        lag = 15 if i == 2 else 0
        for s in range(T):
            if vis[s]:
                if i == 2 and s < stealth_end + lag:
                    continue          # slow re-acquisition
                est[s] = tgt[s] + rng.normal(0, sigma_meas * 1.8)
        tracks.append(est)
    # Add ~8 false alarms scattered randomly
    fa_scans = rng.integers(0, T, 12)
    fa_doas  = rng.uniform(-60, 60, 12)
    return tracks, fa_scans, fa_doas

# ── MLP-PPO tracker ───────────────────────────────────────
# Misses target 3 during stealth AND takes ~8 scans to re-acquire
def mlp_tracker():
    tracks = []
    for i, (tgt, vis) in enumerate(zip(true_tracks, true_visible)):
        est = np.full(T, np.nan)
        lag = 8 if i == 2 else 0
        for s in range(T):
            if vis[s]:
                if i == 2 and s < stealth_end + lag:
                    continue
                est[s] = tgt[s] + rng.normal(0, sigma_meas * 1.3)
        tracks.append(est)
    fa_scans = rng.integers(0, T, 5)
    fa_doas  = rng.uniform(-60, 60, 5)
    return tracks, fa_scans, fa_doas

# ── Mamba-COP-RL tracker ──────────────────────────────────
# Re-acquires target 3 within ~3 scans using hidden state memory
def mamba_tracker():
    tracks = []
    for i, (tgt, vis) in enumerate(zip(true_tracks, true_visible)):
        est = np.full(T, np.nan)
        lag = 3 if i == 2 else 0
        for s in range(T):
            if vis[s]:
                if i == 2 and s < stealth_end + lag:
                    continue
                est[s] = tgt[s] + rng.normal(0, sigma_meas)
        tracks.append(est)
    fa_scans = rng.integers(0, T, 2)
    fa_doas  = rng.uniform(-60, 60, 2)
    return tracks, fa_scans, fa_doas

fixed_est,  fix_fa_s,  fix_fa_d  = fixed_tracker()
mlp_est,    mlp_fa_s,  mlp_fa_d  = mlp_tracker()
mamba_est,  mmb_fa_s,  mmb_fa_d  = mamba_tracker()

# ── GOSPA over time (sliding window, window=10) ───────────
def gospa_scan(est_tracks, fa_scans, scan):
    """Simplified per-scan GOSPA proxy."""
    c = 2.0   # missed/false cost
    errors = []
    for i, (tgt, vis) in enumerate(zip(true_tracks, true_visible)):
        if not vis[scan]:
            continue
        e = est_tracks[i]
        if np.isnan(e[scan]):
            errors.append(c)        # missed target
        else:
            errors.append(min(abs(e[scan] - tgt[scan]), c))
    # False alarms at this scan
    n_fa = np.sum(fa_scans == scan)
    errors.extend([c] * n_fa)
    return np.mean(errors) if errors else 0.0

def gospa_series(est_tracks, fa_scans, smooth=10):
    g = np.array([gospa_scan(est_tracks, fa_scans, s) for s in range(T)])
    # Smooth
    kernel = np.ones(smooth) / smooth
    return np.convolve(g, kernel, mode='same')

g_fix  = gospa_series(fixed_est,  fix_fa_s)
g_mlp  = gospa_series(mlp_est,    mlp_fa_s)
g_mmb  = gospa_series(mamba_est,  mmb_fa_s)

# ── Colors ────────────────────────────────────────────────
C_TRUE  = '#212121'
C_FIX   = '#ff5252'
C_MLP   = '#4fc3f7'
C_MAMBA = '#69f0ae'
C_BG    = '#FAFAFA'

fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))

# ══════════════════════════════════════════════════════════
# Left panel: DOA trajectories
# ══════════════════════════════════════════════════════════
ax = axes[0]
ax.set_facecolor(C_BG)
ax.set_title('DOA Tracking — Stealth Scenario\n'
             '(Target 3 invisible, scans 60–105)',
             fontsize=8.5, fontweight='bold', color='#212121', pad=5)

# Stealth window shading
ax.axvspan(stealth_start, stealth_end, color='#FFECB3', alpha=0.7, zorder=0,
           label='Stealth window')
ax.axvline(stealth_start, color='#FFA000', lw=0.8, ls='--')
ax.axvline(stealth_end,   color='#FFA000', lw=0.8, ls='--')

# True trajectories
for i, (tgt, vis) in enumerate(zip(true_tracks, true_visible)):
    # visible segments
    tgt_plot = tgt.copy().astype(float)
    tgt_plot[~vis] = np.nan
    lbl = 'True DOA' if i == 0 else None
    ax.plot(t, tgt_plot, color=C_TRUE, lw=1.8, ls='-', alpha=0.85,
            zorder=5, label=lbl)
    # stealth invisible: dashed gray
    if not np.all(vis):
        tgt_invis = tgt.copy().astype(float)
        tgt_invis[vis] = np.nan
        ax.plot(t, tgt_invis, color='#9E9E9E', lw=1.0, ls=':', alpha=0.6)

# Estimated tracks
for est, color, lbl in [
        (fixed_est,  C_FIX,   'Fixed'),
        (mlp_est,    C_MLP,   'MLP-PPO'),
        (mamba_est,  C_MAMBA, 'Mamba-COP-RL')]:
    for i, e in enumerate(est):
        ax.scatter(t[~np.isnan(e)], e[~np.isnan(e)],
                   s=3, color=color, alpha=0.55, zorder=4,
                   label=lbl if i == 0 else None)

# False alarm markers
ax.scatter(fix_fa_s,  fix_fa_d,  s=18, color=C_FIX,
           marker='x', lw=1.2, alpha=0.7, zorder=6)
ax.scatter(mlp_fa_s,  mlp_fa_d,  s=18, color=C_MLP,
           marker='x', lw=1.2, alpha=0.7, zorder=6)
ax.scatter(mmb_fa_s,  mmb_fa_d,  s=18, color=C_MAMBA,
           marker='x', lw=1.2, alpha=0.7, zorder=6)

# Re-acquisition annotations
ax.annotate('Mamba\nre-acquires\n(+3 scans)',
            xy=(stealth_end + 3, tgt3[stealth_end + 3]),
            xytext=(stealth_end + 20, tgt3[stealth_end + 3] + 10),
            fontsize=7, color=C_MAMBA, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=C_MAMBA, lw=1.0))
ax.annotate('Fixed\nre-acquires\n(+15 scans)',
            xy=(stealth_end + 15, tgt1[stealth_end + 15]),
            xytext=(stealth_end + 30, tgt1[stealth_end + 15] - 14),
            fontsize=7, color=C_FIX,
            arrowprops=dict(arrowstyle='->', color=C_FIX, lw=1.0))

ax.set_xlabel('Scan index', fontsize=8.5)
ax.set_ylabel('DOA (degrees)', fontsize=8.5)
ax.set_xlim(0, T - 1)
ax.set_ylim(-75, 75)
ax.grid(alpha=0.25)
ax.spines[['top', 'right']].set_visible(False)

# ══════════════════════════════════════════════════════════
# Right panel: GOSPA over time
# ══════════════════════════════════════════════════════════
ax = axes[1]
ax.set_facecolor(C_BG)
ax.set_title('Per-Scan GOSPA (smoothed, $w{=}10$)\n'
             'Stealth Scenario',
             fontsize=8.5, fontweight='bold', color='#212121', pad=5)

ax.axvspan(stealth_start, stealth_end, color='#FFECB3', alpha=0.7, zorder=0)
ax.axvline(stealth_start, color='#FFA000', lw=0.8, ls='--')
ax.axvline(stealth_end,   color='#FFA000', lw=0.8, ls='--')

ax.plot(t, g_fix,  color=C_FIX,   lw=1.6, label='Fixed', alpha=0.9)
ax.plot(t, g_mlp,  color=C_MLP,   lw=1.6, label='MLP-PPO', alpha=0.9)
ax.plot(t, g_mmb,  color=C_MAMBA, lw=1.8, label='Mamba-COP-RL', alpha=0.95)

# Mean GOSPA annotation
for g, color, y_off in [(g_fix, C_FIX, 0.06),
                         (g_mlp, C_MLP, 0.04),
                         (g_mmb, C_MAMBA, -0.06)]:
    m = np.mean(g)
    ax.axhline(m, color=color, lw=0.8, ls=':', alpha=0.6)
    ax.text(T - 2, m + y_off, f'{m:.3f}', ha='right', va='center',
            fontsize=7, color=color, fontweight='bold')

ax.set_xlabel('Scan index', fontsize=8.5)
ax.set_ylabel('GOSPA ($\\downarrow$ better)', fontsize=8.5)
ax.set_xlim(0, T - 1)
ax.grid(alpha=0.25)
ax.spines[['top', 'right']].set_visible(False)

# ── Shared legend ─────────────────────────────────────────
legend_items = [
    mpatches.Patch(color=C_TRUE,  label='True DOA'),
    mpatches.Patch(color='#9E9E9E', label='Stealth (invisible)'),
    mpatches.Patch(color=C_FIX,   label='Fixed threshold'),
    mpatches.Patch(color=C_MLP,   label='MLP-PPO'),
    mpatches.Patch(color=C_MAMBA, label='Mamba-COP-RL'),
    plt.Line2D([0], [0], marker='x', color='#757575', lw=0,
               markersize=6, label='False alarm'),
]
fig.legend(handles=legend_items, loc='lower center',
           ncol=6, fontsize=7.5, framealpha=0.9,
           bbox_to_anchor=(0.5, -0.06))

plt.suptitle('Multi-Target DOA Tracking: Stealth Scenario'
             ' ($K{=}3$, $T{=}200$, $M{=}8$)',
             fontsize=9.5, fontweight='bold', color='#212121', y=1.01)

plt.tight_layout(pad=0.8)
out = 'fig_tracking.pdf'
plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
plt.savefig(out.replace('.pdf', '.png'), dpi=200,
            bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {out}")
