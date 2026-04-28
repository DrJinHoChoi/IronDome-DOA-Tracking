#!/usr/bin/env python3
"""Drone "Eyes and Ears" Animated Demo (per-scan playback).

Renders the same SAR / Combat scenario from drone_demo.py as an MP4
(or GIF fallback) so reviewers can watch the scene unfold scan by scan
instead of inspecting a static 4-panel snapshot.

Layout per frame (gridspec, ~1080p):
  - Top-left  (polar):   live COP polar pseudo-spectrum at the current
                         scan with KWS markers (KW_COLOURS).
  - Top-right (heatmap): COP pseudo-spectrum heatmap revealed up to the
                         current scan; future scans are masked.
  - Bottom    (wide):    GM-PHD tracks revealed up to the current scan,
                         KWS captions appearing the first time a keyword
                         is detected for that track. Vertical "now"
                         cursor.

The animation begins with a 1.5 s intro card (project + footprint
stats) and ends with a 1.5 s outro card (per-track keyword summary).
Real per-scan content runs at 3 fps (one scan per ~333 ms) but every
frame is duplicated so the encoded video plays back at 30 fps for
smoother MP4 output.

Usage:
    python drone_demo_anim.py --scenario sar
    python drone_demo_anim.py --scenario combat --out drone_combat.mp4
    python drone_demo_anim.py --scenario sar --gif      # force GIF

Notes:
    ffmpeg is recommended for MP4 output. If ffmpeg is not installed
    or matplotlib cannot find it, the script automatically falls back
    to a Pillow-based GIF written next to the requested --out path.

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
from matplotlib.animation import FuncAnimation, writers
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from drone_demo import (SCENARIOS, KW_COLOURS, build_trajectory,
                        run_pipeline, attach_kws_labels)


# --------------------------------------------------------------------- #
# Animation parameters                                                  #
# --------------------------------------------------------------------- #
TARGET_FPS = 30           # encoder fps (smooth playback)
SCAN_FPS = 3              # logical scans per second
REPEAT = TARGET_FPS // SCAN_FPS   # frame duplication factor (=10)
INTRO_FRAMES = TARGET_FPS * 3 // 2     # 1.5 s intro card
OUTRO_FRAMES = TARGET_FPS * 3 // 2     # 1.5 s outro card
HOLD_FIRST = TARGET_FPS // 2           # extra hold on scan 0
HOLD_LAST = TARGET_FPS // 2            # extra hold on final scan


# --------------------------------------------------------------------- #
# Frame timeline                                                        #
# --------------------------------------------------------------------- #
def build_timeline(n_scans: int):
    """Return a list of frame descriptors.

    Each entry is a dict: {"kind": "intro"|"scan"|"outro", "scan": int}.
    Scans are duplicated REPEAT times for smooth 30 fps playback, with
    additional hold frames at the start/end of the scan portion.
    """
    timeline = []
    for _ in range(INTRO_FRAMES):
        timeline.append({"kind": "intro", "scan": 0})
    for s in range(n_scans):
        reps = REPEAT
        if s == 0:
            reps += HOLD_FIRST
        if s == n_scans - 1:
            reps += HOLD_LAST
        for _ in range(reps):
            timeline.append({"kind": "scan", "scan": s})
    for _ in range(OUTRO_FRAMES):
        timeline.append({"kind": "outro", "scan": n_scans - 1})
    return timeline


# --------------------------------------------------------------------- #
# Renderer                                                              #
# --------------------------------------------------------------------- #
class AnimRenderer:
    def __init__(self, history, labelled, scenario):
        self.history = history
        self.labelled = labelled
        self.scenario = scenario
        self.n_scans = scenario["n_scans"]
        self.spectra = np.stack(history["spectra"])  # (n_scans, 181)
        self.angles = history["scan_angles_deg"]
        self.gt = history["gt"]

        # Per-track point lists (built incrementally during animation).
        self.track_xy = {}
        for s, per_scan in enumerate(labelled):
            for tid, deg, kw in per_scan:
                self.track_xy.setdefault(tid, []).append((s, deg, kw))

        # Figure layout (~1080p at dpi=120 -> 16x9 -> 1920x1080).
        self.fig = plt.figure(figsize=(16, 9), dpi=120)
        gs = self.fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05],
                                   hspace=0.42, wspace=0.22,
                                   left=0.06, right=0.97,
                                   top=0.90, bottom=0.08)
        self.ax_polar = self.fig.add_subplot(gs[0, 0], projection="polar")
        self.ax_heat = self.fig.add_subplot(gs[0, 1])
        self.ax_track = self.fig.add_subplot(gs[1, :])

        self.suptitle = self.fig.suptitle(
            f"Drone Eyes & Ears -- {scenario['name']} "
            f"(8-mic ULA, NC-SSM/NC-TCN + Mamba-COP-RFS)",
            fontsize=15, fontweight="bold")

        # Pre-build heatmap canvas (will mask future scans per frame).
        proxy = 20.0 * np.log10(np.abs(self.spectra) + 1e-3)
        self.heat_full = proxy.T  # (181, n_scans)
        self.heat_vmin = float(np.nanmin(self.heat_full))
        self.heat_vmax = float(np.nanmax(self.heat_full))

        # Track palette
        self.palette = plt.cm.tab10(np.linspace(0, 1, 10))

        # Footprint stats (used in intro)
        self.n_sources = len(scenario["sources"])
        self.snr_db = scenario["snr_db"]

    # ----------------------------- helpers ----------------------------
    def _clear_axes(self):
        for ax in (self.ax_polar, self.ax_heat, self.ax_track):
            ax.clear()

    def _draw_scan(self, s):
        """Render a real scan frame at index s."""
        self._clear_axes()

        # ---- Polar ----
        ax = self.ax_polar
        spec = self.spectra[s]
        norm = spec / (spec.max() + 1e-12)
        ax.plot(np.deg2rad(self.angles), norm,
                color="#2C3E50", lw=1.6)
        ax.fill_between(np.deg2rad(self.angles), 0, norm,
                        color="#2C3E50", alpha=0.10)
        for tid, deg, kw in self.labelled[s]:
            col = KW_COLOURS.get(kw, "#888888")
            ax.plot([np.deg2rad(deg)], [1.0], 'o', color=col, ms=12,
                    markeredgecolor="black", markeredgewidth=1.2)
            ax.text(np.deg2rad(deg), 1.18, kw or "?", color=col,
                    fontsize=10, fontweight="bold", ha="center")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_thetamin(-90)
        ax.set_thetamax(90)
        ax.set_ylim(0, 1.3)
        ax.set_yticklabels([])
        ax.set_title(f"Polar DOA + KWS  |  scan {s+1}/{self.n_scans}",
                     fontsize=12, fontweight="bold", pad=14)

        # ---- Heatmap (mask future scans) ----
        ax = self.ax_heat
        masked = np.full_like(self.heat_full, np.nan, dtype=float)
        masked[:, : s + 1] = self.heat_full[:, : s + 1]
        ax.imshow(masked, aspect="auto", origin="lower",
                  extent=[0, self.n_scans,
                          self.angles[0], self.angles[-1]],
                  cmap="magma",
                  vmin=self.heat_vmin, vmax=self.heat_vmax)
        ax.axvline(s + 0.5, color="white", lw=1.3, alpha=0.85)
        ax.set_title("COP pseudo-spectrum (revealed)",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Scan index")
        ax.set_ylabel("DOA (deg)")
        ax.set_xlim(0, self.n_scans)

        # ---- Tracks (revealed up to s) ----
        ax = self.ax_track
        # Ground truth dashed (full, faint).
        for k in range(self.gt.shape[1]):
            ax.plot(np.arange(self.n_scans), self.gt[:, k],
                    color="black", lw=0.7, linestyle="--",
                    alpha=0.30,
                    label="ground truth" if k == 0 else None)

        # Occlusion shading (full).
        first_occl_src = (self.scenario["occlusions"][0][0]
                          if self.scenario["occlusions"] else None)
        for src_idx, a, b in self.scenario["occlusions"]:
            ax.axvspan(a, b, alpha=0.15, color="#F39C12",
                       label="occlusion" if src_idx == first_occl_src
                       else None)

        # Tracks up to current scan.
        for tid, pts in self.track_xy.items():
            visible = [p for p in pts if p[0] <= s]
            if not visible:
                continue
            ss = [p[0] for p in visible]
            ds = [p[1] for p in visible]
            ax.plot(ss, ds, '-', color=self.palette[tid % 10], lw=2,
                    marker='o', ms=5, label=f"track #{tid}")
            # KWS captions on first appearance / change.
            prev_kw = None
            for sc, d, kw in visible:
                if kw and kw != prev_kw:
                    col = KW_COLOURS.get(kw, "#888888")
                    ax.annotate(kw, xy=(sc, d), xytext=(sc, d + 8),
                                fontsize=9, color=col, fontweight="bold",
                                ha="center",
                                arrowprops=dict(arrowstyle="->",
                                                color=col, lw=0.8))
                    prev_kw = kw

        # "Now" cursor.
        ax.axvline(s, color="#34495E", lw=1.4, alpha=0.7)
        ax.set_title("GM-PHD tracks (Mamba-COP-RFS) + KWS captions "
                     "(NC-SSM/NC-TCN)",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Scan index")
        ax.set_ylabel("DOA (deg)")
        ax.set_xlim(-0.5, self.n_scans - 0.5)
        ax.set_ylim(self.angles[0], self.angles[-1])
        ax.legend(loc="upper right", fontsize=9, ncol=2, framealpha=0.9)
        ax.grid(alpha=0.25)

    def _draw_intro(self):
        self._clear_axes()
        for ax in (self.ax_polar, self.ax_heat, self.ax_track):
            ax.set_axis_off()
        # Use ax_track as the main canvas.
        ax = self.ax_track
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.text(0.5, 0.78,
                "Drone Eyes & Ears -- Integrated Demo",
                ha="center", va="center",
                fontsize=26, fontweight="bold", color="#2C3E50")
        ax.text(0.5, 0.62,
                f"Scenario: {self.scenario['name']}",
                ha="center", va="center",
                fontsize=18, color="#34495E")
        stats = (f"{self.n_scans} scans  |  {self.n_sources} sources  |  "
                 f"8-mic ULA  |  SNR {self.snr_db:.0f} dB")
        ax.text(0.5, 0.50, stats,
                ha="center", va="center",
                fontsize=14, color="#2C3E50")
        ax.text(0.5, 0.36,
                "NC-SSM / NC-TCN (KWS)  +  Mamba-COP-RFS (GM-PHD DOA)",
                ha="center", va="center",
                fontsize=13, color="#7F8C8D", style="italic")
        ax.text(0.5, 0.20,
                "Per-scan playback  ->  starting...",
                ha="center", va="center",
                fontsize=12, color="#95A5A6")

    def _draw_outro(self):
        self._clear_axes()
        for ax in (self.ax_polar, self.ax_heat, self.ax_track):
            ax.set_axis_off()
        ax = self.ax_track
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.text(0.5, 0.86,
                f"Summary -- {self.scenario['name']}",
                ha="center", va="center",
                fontsize=22, fontweight="bold", color="#2C3E50")
        # Per-track keyword summary
        lines = []
        for tid, pts in sorted(self.track_xy.items()):
            kws = []
            for _, _, kw in pts:
                if kw and (not kws or kws[-1] != kw):
                    kws.append(kw)
            if not kws:
                kws = ["(no keyword)"]
            lines.append(f"track #{tid}:  " + ", ".join(kws))
        y = 0.70
        for line in lines:
            ax.text(0.5, y, line,
                    ha="center", va="center",
                    fontsize=13, color="#34495E")
            y -= 0.07
        ax.text(0.5, 0.10,
                f"{len(self.track_xy)} tracks confirmed across "
                f"{self.n_scans} scans.",
                ha="center", va="center",
                fontsize=12, color="#7F8C8D", style="italic")

    # ------------------------------ API -------------------------------
    def update(self, frame_desc):
        kind = frame_desc["kind"]
        if kind == "intro":
            self._draw_intro()
        elif kind == "outro":
            self._draw_outro()
        else:
            self._draw_scan(frame_desc["scan"])
        return []


# --------------------------------------------------------------------- #
# Driver                                                                #
# --------------------------------------------------------------------- #
def write_animation(renderer: AnimRenderer, timeline, out_path: Path,
                    fps: int, force_gif: bool):
    """Save the animation, falling back to GIF if ffmpeg is unavailable."""
    anim = FuncAnimation(renderer.fig, renderer.update,
                         frames=timeline, interval=1000.0 / fps,
                         blit=False, repeat=False)

    can_ffmpeg = writers.is_available("ffmpeg")
    use_gif = force_gif or (not can_ffmpeg) or \
        out_path.suffix.lower() == ".gif"

    if use_gif:
        gif_path = (out_path.with_suffix(".gif")
                    if out_path.suffix.lower() != ".gif" else out_path)
        if not force_gif and not can_ffmpeg:
            print("[WARN] ffmpeg writer not available; "
                  "falling back to GIF.")
        # GIF can be slow at 30 fps; cap it for sane file sizes.
        gif_fps = min(fps, 15)
        print(f"[..] writing GIF -> {gif_path} (fps={gif_fps}, "
              f"frames={len(timeline)})")
        anim.save(str(gif_path), writer="pillow", fps=gif_fps,
                  dpi=100)
        print(f"[OK] saved {gif_path}")
        return gif_path
    else:
        print(f"[..] writing MP4 -> {out_path} (fps={fps}, "
              f"frames={len(timeline)})")
        anim.save(str(out_path), writer="ffmpeg", fps=fps,
                  dpi=120,
                  extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])
        print(f"[OK] saved {out_path}")
        return out_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=list(SCENARIOS), default="sar")
    ap.add_argument("--out", type=Path, default=REPO / "drone_demo.mp4")
    ap.add_argument("--fps", type=int, default=TARGET_FPS)
    ap.add_argument("--gif", action="store_true",
                    help="Force GIF output even if ffmpeg is available.")
    ap.add_argument("--max-scans", type=int, default=None,
                    help="Truncate scenario to N scans (testing).")
    args = ap.parse_args()

    scenario = dict(SCENARIOS[args.scenario])  # shallow copy
    if args.max_scans is not None:
        n = min(args.max_scans, scenario["n_scans"])
        scenario["n_scans"] = n
        # Clamp source/occlusion ranges so build_trajectory stays in bounds.
        scenario["sources"] = [
            (min(t_in, n), min(t_out, n), th0, th1, kw)
            for (t_in, t_out, th0, th1, kw) in scenario["sources"]
            if t_in < n
        ]
        scenario["occlusions"] = [
            (idx, min(a, n), min(b, n))
            for (idx, a, b) in scenario["occlusions"]
            if a < n
        ]

    print(f"=== {scenario['name']} ({scenario['n_scans']} scans, "
          f"{len(scenario['sources'])} sources) ===")
    t0 = time.time()
    history = run_pipeline(scenario, use_kws=True)
    labelled = attach_kws_labels(history, scenario)
    print(f"[..] pipeline elapsed: {time.time() - t0:.1f} s")

    renderer = AnimRenderer(history, labelled, scenario)
    timeline = build_timeline(scenario["n_scans"])
    print(f"[..] total frames: {len(timeline)} "
          f"(intro={INTRO_FRAMES}, outro={OUTRO_FRAMES}, "
          f"scans={scenario['n_scans']} x ~{REPEAT})")

    write_animation(renderer, timeline, args.out, args.fps, args.gif)
    plt.close(renderer.fig)

    print(f"total elapsed: {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
