#!/usr/bin/env python3
"""Live laptop-camera + NC-Conv-SSM Vision demo.

Captures frames from the laptop webcam, runs an NC-Conv classifier
(loaded from the NC-SSM-Vision-ICCV2027 sibling repo) and overlays
the prediction + sigma-gate diagnostic on the live feed.

Pairs with `drone_demo_mic.py` to give the full eyes-and-ears stack
on a single laptop:
    eyes  -> drone_demo_vision.py  (NC-Conv-SSM)
    ears  -> drone_demo_mic.py     (NC-SSM / NC-TCN KWS)

Sibling repo (override via NC_VISION_REPO env var):
    default: ../NC-SSM-Vision-ICCV2027

Usage:
    python drone_demo_vision.py                # live webcam
    python drone_demo_vision.py --camera 1     # specific cam index
    python drone_demo_vision.py --test 2.0     # 2-second capture, exit
    python drone_demo_vision.py --image FILE   # single-image inference

Requirements:
    pip install opencv-python torch torchvision pillow

License: see LICENSE / COMMERCIAL_LICENSE.md.
Author: Jin Ho Choi (SmartEAR / NanoAgentic AI).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent

# Vision repo / package location
_env = os.environ.get("NC_VISION_REPO", "").strip()
VISION_PARENTS = []
if _env:
    VISION_PARENTS.append(Path(_env))
VISION_PARENTS += [
    REPO.parent / "NC-SSM-Vision-ICCV2027",
    REPO.parent / "NanoMamba-Interspeech2026",
]
VISION_PARENTS = [p for p in VISION_PARENTS if p.is_dir()]


# CIFAR-10 labels (default classifier head)
CIFAR_LABELS = ["airplane", "automobile", "bird", "cat", "deer",
                "dog", "frog", "horse", "ship", "truck"]


def load_ncconv_classifier(n_classes: int = 10):
    """Try to load NC-Conv classifier model. Returns (model, status_str).

    Falls back to randomly-initialized model so the demo runs even
    without a checkpoint -- useful for showing the architecture
    footprint and the live sigma-gate diagnostic.
    """
    if not VISION_PARENTS:
        return None, "[vision] sibling repo not found; set NC_VISION_REPO"

    parent = VISION_PARENTS[0]
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))
    try:
        import torch
        from ncconv.models import NCConvBlock, make_ncconv_net
    except Exception as e:
        return None, f"[vision] import failed: {e}"

    net = make_ncconv_net(NCConvBlock, n_classes=n_classes)
    n_params = sum(p.numel() for p in net.parameters())
    net.eval()

    # Try to find a checkpoint -- official path is variable; we accept
    # any *.pt under checkpoints_vision/.
    ckpt_dir = parent / "checkpoints_vision"
    ckpt = None
    if ckpt_dir.is_dir():
        cands = list(ckpt_dir.glob("ncconv_*.pt"))
        if cands:
            ckpt = cands[0]
    if ckpt is None and (parent / "ncconv" / "best.pt").is_file():
        ckpt = parent / "ncconv" / "best.pt"

    if ckpt is not None:
        try:
            state = torch.load(ckpt, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            net.load_state_dict(state, strict=False)
            return (net,
                    f"[vision] NC-Conv loaded from {ckpt.name} "
                    f"({n_params:,} params, {n_params/1024:.1f} KB INT8)")
        except Exception as e:
            return (net,
                    f"[vision] NC-Conv built (random init -- ckpt load "
                    f"failed: {e}; {n_params:,} params)")
    return (net,
            f"[vision] NC-Conv built (random init -- no checkpoint; "
            f"{n_params:,} params, {n_params/1024:.1f} KB INT8)")


# --------------------------------------------------------------------- #
# Inference helpers                                                     #
# --------------------------------------------------------------------- #
def preprocess_frame(frame_bgr: np.ndarray, size: int = 32) -> "torch.Tensor":
    """BGR uint8 frame -> (1, 3, size, size) float tensor in [0, 1]."""
    import cv2
    import torch
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    arr = rgb.astype("float32") / 255.0
    arr = arr.transpose(2, 0, 1)[None, ...]
    return torch.from_numpy(arr)


def classify(net, frame_bgr: np.ndarray, labels=CIFAR_LABELS):
    import torch
    x = preprocess_frame(frame_bgr)
    with torch.no_grad():
        logits = net(x)
        probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()
    idx = int(np.argmax(probs))
    return labels[idx % len(labels)], float(probs[idx])


def estimate_sigma(frame_bgr: np.ndarray) -> float:
    """Crude noise / contrast estimator -- proxy for the NC-Conv
    sigma gate. Returns dimensionless 0..1."""
    import cv2
    g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype("float32")
    return float(np.clip(np.std(g) / 80.0, 0.0, 1.0))


# --------------------------------------------------------------------- #
# Live webcam                                                           #
# --------------------------------------------------------------------- #
def run_webcam(args, net) -> int:
    try:
        import cv2
    except ImportError:
        print("[ERROR] opencv-python missing. pip install opencv-python")
        return 2

    cam = cv2.VideoCapture(args.camera)
    if not cam.isOpened():
        print(f"[ERROR] cannot open camera {args.camera}")
        return 4

    print("[live] press 'q' to quit, 's' to save snapshot.")
    last_pred = ("...", 0.0)
    last_inf = 0.0
    frame_idx = 0

    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                print("[ERROR] frame grab failed")
                break
            frame_idx += 1
            sigma = estimate_sigma(frame)

            # Throttle inference to ~5 Hz
            now = time.time()
            if now - last_inf > 0.2:
                last_pred = classify(net, frame)
                last_inf = now

            label, conf = last_pred
            txt1 = f"NC-Conv: {label} ({conf*100:.0f}%)"
            txt2 = f"sigma_gate: {sigma:.2f}"
            cv2.putText(frame, txt1, (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, txt2, (12, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            cv2.imshow("Drone Eyes -- NC-Conv-SSM", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                snap = REPO / f"vision_snap_{frame_idx:05d}.png"
                cv2.imwrite(str(snap), frame)
                print(f"  saved {snap.name}")
    finally:
        cam.release()
        cv2.destroyAllWindows()
    return 0


# --------------------------------------------------------------------- #
# Test / single-image                                                   #
# --------------------------------------------------------------------- #
def run_test(args, net) -> int:
    try:
        import cv2
    except ImportError:
        print("[ERROR] opencv-python missing. pip install opencv-python")
        return 2
    cam = cv2.VideoCapture(args.camera)
    if not cam.isOpened():
        print(f"[ERROR] cannot open camera {args.camera}")
        return 4
    print(f"[test] capturing {args.test:.1f} s on camera {args.camera} ...")
    t0 = time.time()
    frames = 0
    last_label = None
    while time.time() - t0 < args.test:
        ok, frame = cam.read()
        if not ok:
            break
        frames += 1
        last_frame = frame
    cam.release()
    if frames == 0:
        print("[ERROR] no frames captured")
        return 4
    label, conf = classify(net, last_frame)
    sigma = estimate_sigma(last_frame)
    print(f"[test] {frames} frames in {time.time()-t0:.1f} s, "
          f"frame_size={last_frame.shape}")
    print(f"[test] sigma_gate={sigma:.2f}  NC-Conv -> {label} ({conf:.2f})")
    return 0


def run_image(args, net) -> int:
    try:
        import cv2
    except ImportError:
        print("[ERROR] opencv-python missing. pip install opencv-python")
        return 2
    img = cv2.imread(str(args.image))
    if img is None:
        print(f"[ERROR] cannot read image {args.image}")
        return 4
    label, conf = classify(net, img)
    sigma = estimate_sigma(img)
    print(f"[image] {args.image}  size={img.shape}  sigma={sigma:.2f}")
    print(f"[image] NC-Conv -> {label} ({conf:.2f})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--camera", type=int, default=0,
                    help="Camera index (default 0).")
    ap.add_argument("--test", type=float, default=None,
                    help="Test capture: grab N seconds, classify last "
                         "frame, exit (no GUI).")
    ap.add_argument("--image", type=Path, default=None,
                    help="Run inference on a single image and exit.")
    args = ap.parse_args()

    net, status = load_ncconv_classifier()
    print(status)
    if net is None:
        return 3

    if args.image:
        return run_image(args, net)
    if args.test is not None:
        return run_test(args, net)
    return run_webcam(args, net)


if __name__ == "__main__":
    sys.exit(main())
