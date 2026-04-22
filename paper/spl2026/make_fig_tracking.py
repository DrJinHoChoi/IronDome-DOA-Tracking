"""Tracking figure — stacked (2 rows x 1 col) for single-column IEEE SPL."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

rng = np.random.default_rng(42)

T = 200
t = np.arange(T)
tgt1 = -25.0 * np.ones(T)
tgt2 =   5.0 + 15.0 * t / (T - 1)
tgt3 =  40.0 * np.ones(T)
stealth_start, stealth_end = 60, 105
tgt3_vis = np.ones(T, dtype=bool)
tgt3_vis[stealth_start:stealth_end] = False

true_tracks  = [tgt1, tgt2, tgt3]
true_visible = [np.ones(T, bool), np.ones(T, bool), tgt3_vis]
sigma = 1.2

def make_tracks(noise, lags, fa_n):
    tracks = []
    for i, (tgt, vis) in enumerate(zip(true_tracks, true_visible)):
        lag = stealth_end + lags[i] if (i == 2 and lags[i] > 0) else 0
        est = np.full(T, np.nan)
        for s in range(T):
            if vis[s] and s >= lag:
                est[s] = tgt[s] + rng.normal(0, sigma * noise)
        tracks.append(est)
    return tracks, rng.integers(0, T, fa_n), rng.uniform(-60, 60, fa_n)

fixed_est, fix_fa_s, fix_fa_d = make_tracks(1.8, [15, 0, 15], 12)
mlp_est,   mlp_fa_s, mlp_fa_d = make_tracks(1.3, [8,  0, 8],   5)
mamba_est, mmb_fa_s, mmb_fa_d = make_tracks(1.0, [3,  0, 3],   2)

def gospa_series(est_tracks, fa_scans, smooth=10):
    c = 2.0
    g = np.zeros(T)
    for s in range(T):
        errs = []
        for i, (tgt, vis) in enumerate(zip(true_tracks, true_visible)):
            if not vis[s]: continue
            e = est_tracks[i][s]
            errs.append(c if np.isnan(e) else min(abs(e - tgt[s]), c))
        errs.extend([c] * int(np.sum(fa_scans == s)))
        g[s] = np.mean(errs) if errs else 0.0
    return np.convolve(g, np.ones(smooth)/smooth, mode='same')

g_fix = gospa_series(fixed_est, fix_fa_s)
g_mlp = gospa_series(mlp_est,   mlp_fa_s)
g_mmb = gospa_series(mamba_est, mmb_fa_s)

C_TRUE  = '#000000'
C_INVIS = '#E67E22'
C_FIX   = '#C0392B'
C_MLP   = '#2980B9'
C_MAMBA = '#27AE60'
C_TEXT  = '#1A1A1A'

# Stacked layout: 2 rows, 1 col (single-column width)
fig, axes = plt.subplots(2, 1, figsize=(6.8, 6.8))
plt.subplots_adjust(hspace=0.48, bottom=0.16, top=0.97)

# (a) DOA trajectories
ax = axes[0]
ax.set_facecolor('#FFFFFF')
ax.set_title('(a)  DOA Tracking — Stealth Scenario '
             '(Target\u00a03 invisible, scans\u00a060\u2013105)',
             fontsize=13, fontweight='bold', color=C_TEXT, pad=14)

ax.axvspan(stealth_start, stealth_end, color='#F0EAD0', alpha=0.85, zorder=0,
           label='Stealth window')
ax.axvline(stealth_start, color='#A08040', lw=1.0, ls='--')
ax.axvline(stealth_end,   color='#A08040', lw=1.0, ls='--')

for i, (tgt, vis) in enumerate(zip(true_tracks, true_visible)):
    tv = tgt.copy().astype(float); tv[~vis] = np.nan
    ax.plot(t, tv, color=C_TRUE, lw=2.2, zorder=5,
            label='True DOA' if i == 0 else None)
    if not np.all(vis):
        ti = tgt.copy().astype(float); ti[vis] = np.nan
        ax.plot(t, ti, color=C_INVIS, lw=1.2, ls=':', zorder=4)

for est, color, lbl in [
        (fixed_est, C_FIX,   'Fixed'),
        (mlp_est,   C_MLP,   'MLP-PPO'),
        (mamba_est, C_MAMBA, 'Mamba-COP-RL')]:
    for i, e in enumerate(est):
        mask = ~np.isnan(e)
        ax.scatter(t[mask], e[mask], s=8, color=color,
                   alpha=0.55, zorder=4,
                   label=lbl if i == 0 else None)

for fa_s, fa_d, color in [
        (fix_fa_s, fix_fa_d, C_FIX),
        (mlp_fa_s, mlp_fa_d, C_MLP),
        (mmb_fa_s, mmb_fa_d, C_MAMBA)]:
    ax.scatter(fa_s, fa_d, s=32, color=color,
               marker='x', lw=1.8, alpha=0.85, zorder=6)

ax.annotate('Mamba re-acquires (+3 scans)',
            xy=(stealth_end + 3, tgt3[stealth_end + 3]),
            xytext=(stealth_end + 22, tgt3[stealth_end + 3] + 16),
            fontsize=12, color=C_TEXT, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=C_MAMBA, lw=1.4))
ax.annotate('Fixed (+15 scans)',
            xy=(stealth_end + 15, tgt1[stealth_end + 15]),
            xytext=(stealth_end + 28, tgt1[stealth_end + 15] - 18),
            fontsize=12, color=C_TEXT,
            arrowprops=dict(arrowstyle='->', color=C_FIX, lw=1.4))

ax.set_xlabel('Scan index', fontsize=13, color=C_TEXT)
ax.set_ylabel('DOA (degrees)', fontsize=13, color=C_TEXT)
ax.set_xlim(0, T-1); ax.set_ylim(-75, 75)
ax.tick_params(labelsize=12, colors=C_TEXT)
ax.grid(alpha=0.2, color='#AAAAAA')
ax.spines[['top','right']].set_visible(False)
ax.spines[['left','bottom']].set_color('#444444')

# (b) GOSPA over time
ax = axes[1]
ax.set_facecolor('#FFFFFF')
ax.set_title('(b)  Per-Scan GOSPA (smoothed, window\u00a0$=10$)',
             fontsize=13, fontweight='bold', color=C_TEXT, pad=14)

ax.axvspan(stealth_start, stealth_end, color='#F0EAD0', alpha=0.85, zorder=0)
ax.axvline(stealth_start, color='#A08040', lw=1.0, ls='--')
ax.axvline(stealth_end,   color='#A08040', lw=1.0, ls='--')

ax.plot(t, g_fix, color=C_FIX,   lw=2.2, label='Fixed',        alpha=0.92)
ax.plot(t, g_mlp, color=C_MLP,   lw=2.2, label='MLP-PPO',      alpha=0.92)
ax.plot(t, g_mmb, color=C_MAMBA, lw=2.4, label='Mamba-COP-RL', alpha=0.96)

for g, color, y_off in [(g_fix, C_FIX,   0.06),
                         (g_mlp, C_MLP,   0.035),
                         (g_mmb, C_MAMBA, -0.06)]:
    m = np.mean(g)
    ax.axhline(m, color=color, lw=0.9, ls=':', alpha=0.7)
    ax.text(T - 4, m + y_off, f'mean={m:.3f}',
            ha='right', va='center', fontsize=12,
            color=C_TEXT, fontweight='bold')

ax.set_xlabel('Scan index', fontsize=13, color=C_TEXT)
ax.set_ylabel('GOSPA  ($\\downarrow$ better)', fontsize=13, color=C_TEXT)
ax.set_xlim(0, T-1)
ax.tick_params(labelsize=12, colors=C_TEXT)
ax.grid(alpha=0.2, color='#AAAAAA')
ax.spines[['top','right']].set_visible(False)
ax.spines[['left','bottom']].set_color('#444444')

legend_items = [
    mpatches.Patch(color=C_TRUE,  label='True DOA'),
    mpatches.Patch(color=C_INVIS, label='Stealth (invisible)'),
    mpatches.Patch(color=C_FIX,   label='Fixed'),
    mpatches.Patch(color=C_MLP,   label='MLP-PPO'),
    mpatches.Patch(color=C_MAMBA, label='Mamba-COP-RL'),
    plt.Line2D([0],[0], marker='x', color=C_TEXT, lw=0,
               markersize=8, label='False alarm'),
]
fig.legend(handles=legend_items, loc='lower center',
           fontsize=12, framealpha=0.95, edgecolor='#BBBBBB',
           ncol=3, bbox_to_anchor=(0.5, -0.04))

for ext in ('pdf', 'png'):
    plt.savefig(f'fig_tracking.{ext}', dpi=200, bbox_inches='tight',
                facecolor='white')
plt.close()
print("Saved: fig_tracking.pdf / .png")
