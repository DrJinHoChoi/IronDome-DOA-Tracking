#!/usr/bin/env python3
"""Live laptop-microphone KWS demo (no array, no DOA).

Captures 1-second windows from the laptop's default microphone, runs
NC-SSM (or NC-TCN) keyword spotting, and prints the detected keyword
in real time. Use this as the "ears only" subset of the full drone
eyes-and-ears demo until a multi-mic array (e.g. ReSpeaker) is wired.

Limitations vs full drone demo (drone_demo.py):
  - 1 channel -> NO direction-of-arrival estimation. The DOA is faked
    using a slow virtual-circle trajectory just for visualisation.
  - GM-PHD is bypassed (no multi-target tracking on a single channel).
  - The keyword set is restricted to GSC v0.02 12-class. Out-of-vocab
    voices are reported as "unknown".

Usage:
    python drone_demo_mic.py                       # NC-SSM-Tiny live
    python drone_demo_mic.py --model nc-tcn        # NC-TCN-20K
    python drone_demo_mic.py --backbone Small      # NC-SSM-Small
    python drone_demo_mic.py --device 1            # specific input device
    python drone_demo_mic.py --list-devices        # show available mics

Requirements:
    pip install sounddevice numpy
    (drone_demo_kws.py uses torch + the NC-SSM/NC-TCN sibling repos)

License: see LICENSE / COMMERCIAL_LICENSE.md.
Author: Jin Ho Choi (SmartEAR / NanoAgentic AI).
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from drone_demo_kws import RealKWSClassifier, GSC_LABELS

SR = 16000
WIN = 16000          # 1 s window
HOP = 4000           # 0.25 s hop -> 4 detections / second
RMS_THRESHOLD = 0.005  # below this, we report silence and skip inference

# Many Windows laptops ship a 4-channel Intel Smart Sound array; capturing
# all channels gives a (rough) hint at DOA via inter-channel level ratios.
DEFAULT_CHANNELS = 1


def list_devices() -> None:
    import sounddevice as sd
    print(sd.query_devices())


def color_for(label: str) -> str:
    """ANSI colour for a predicted keyword (kept ASCII-friendly)."""
    palette = {
        "yes":  "\x1b[92m", "no":  "\x1b[91m",
        "up":   "\x1b[93m", "down": "\x1b[94m",
        "left": "\x1b[96m", "right": "\x1b[95m",
        "on":   "\x1b[92m", "off":   "\x1b[91m",
        "stop": "\x1b[91m", "go":    "\x1b[92m",
        "silence": "\x1b[90m", "unknown": "\x1b[37m",
    }
    return palette.get(label, "\x1b[37m")


def run_test_capture(args, sd) -> int:
    """Non-interactive capture test: grab N seconds, run KWS once,
    report per-channel RMS + prediction. Useful for CI / smoke tests."""
    sec = args.test
    n = int(sec * SR)
    print(f"[test] capturing {sec:.1f} s on device={args.device} "
          f"channels={args.channels} ...")
    try:
        rec = sd.rec(n, samplerate=SR, channels=args.channels,
                     device=args.device, dtype="float32", blocking=True)
    except Exception as e:
        print(f"[ERROR] capture failed: {e}")
        return 4
    if rec.ndim == 1:
        rec = rec[:, None]
    rms_per = np.sqrt(np.mean(rec ** 2, axis=0) + 1e-12)
    print(f"[test] captured shape={rec.shape}  "
          f"per-channel RMS={['%.4f' % r for r in rms_per]}")
    if args.channels > 1:
        # Crude L/R cue from level imbalance (2-mic) or front/back
        # variance (4-mic). Real DOA needs ULA geometry knowledge.
        ratio = rms_per / (rms_per.max() + 1e-9)
        print(f"[test] level ratio (max=1) {['%.2f' % r for r in ratio]}")
    mono = rec.mean(axis=1)
    if mono.shape[0] >= WIN:
        wav = mono[-WIN:]
    else:
        wav = np.pad(mono, (0, WIN - mono.shape[0]))
    kws = RealKWSClassifier(model=args.model, backbone=args.backbone)
    print(kws.status())
    if not kws.ok:
        return 3
    label, conf = kws.classify(wav)
    print(f"[test] KWS -> {label}  conf={conf:.2f}")
    return 0


def run(args) -> int:
    try:
        import sounddevice as sd
    except ImportError:
        print("[ERROR] sounddevice not installed.\n"
              "        pip install sounddevice")
        return 2

    if args.list_devices:
        list_devices()
        return 0

    if args.test is not None:
        return run_test_capture(args, sd)

    kws = RealKWSClassifier(model=args.model, backbone=args.backbone)
    print(kws.status())
    if not kws.ok:
        print("[fatal] cannot load KWS model -- exiting.")
        return 3

    print(f"[live] sample_rate={SR} window={WIN/SR:.2f}s "
          f"hop={HOP/SR:.2f}s threshold_rms={RMS_THRESHOLD}")
    print("       press Ctrl+C to stop.\n")

    buf = deque(maxlen=WIN)
    buf.extend(np.zeros(WIN, dtype="float32"))
    last_label = None
    last_time = 0.0

    def cb(indata, frames, t_, status):
        if status:
            print(f"[stream] {status}", file=sys.stderr)
        # mono mix
        x = indata[:, 0] if indata.ndim > 1 else indata
        buf.extend(x.astype("float32"))

    try:
        with sd.InputStream(device=args.device, channels=1,
                            samplerate=SR, blocksize=HOP,
                            callback=cb):
            while True:
                time.sleep(HOP / SR)
                wav = np.asarray(buf, dtype="float32")
                rms = float(np.sqrt(np.mean(wav ** 2) + 1e-12))
                if rms < RMS_THRESHOLD:
                    if last_label != "silence":
                        print(f"\r{color_for('silence')}silence "
                              f"\x1b[0m(rms={rms:.4f})    ",
                              end="", flush=True)
                    last_label = "silence"
                    continue
                label, conf = kws.classify(wav)
                # rate-limit identical predictions
                if (label == last_label and
                        time.time() - last_time < 0.6):
                    continue
                last_time = time.time()
                last_label = label
                stamp = time.strftime("%H:%M:%S")
                print(f"\r{color_for(label)}{stamp}  "
                      f"{label:<10}\x1b[0m  "
                      f"conf={conf:.2f}  rms={rms:.3f}",
                      flush=True)
                print()
    except KeyboardInterrupt:
        print("\n[stopped]")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", choices=["nc-ssm", "nc-tcn"], default="nc-ssm")
    ap.add_argument("--backbone", choices=["Tiny", "Small"], default="Tiny",
                    help="NC-SSM only.")
    ap.add_argument("--device", type=int, default=None,
                    help="Input device index. Default: system default.")
    ap.add_argument("--list-devices", action="store_true",
                    help="List available audio devices and exit.")
    ap.add_argument("--channels", type=int, default=DEFAULT_CHANNELS,
                    help="Number of input channels to capture.")
    ap.add_argument("--test", type=float, default=None,
                    help="Test capture: grab N seconds, run KWS once, "
                         "exit (no live loop). Useful for CI.")
    args = ap.parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
