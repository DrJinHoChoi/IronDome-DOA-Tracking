"""Real NC-SSM / NC-TCN KWS classifier wrapper for drone_demo.

Provides `RealKWSClassifier` that loads a pre-trained NC-SSM-Tiny or
NC-TCN-20K checkpoint from a sibling repo and classifies a 1-second
16 kHz mono waveform.

If sibling repos / checkpoints are missing, the loader prints a clear
diagnostic and the demo falls back to the ground-truth stub.

Sibling repos searched (override via env vars):
    NC_SSM_REPO   default: ../NC-SSM-TASLP2026
    NC_TCN_REPO   default: ../NC-TCN

Example:
    >>> kws = RealKWSClassifier(model='nc-ssm', backbone='Tiny')
    >>> kws.classify(np.random.randn(16000))
    ('unknown', 0.34)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

REPO = Path(__file__).resolve().parent

# Standard 12-class GSC v0.02 label set (10 keywords + silence + unknown)
GSC_LABELS = ["yes", "no", "up", "down", "left", "right",
              "on", "off", "stop", "go", "silence", "unknown"]

# Map demo keywords (sar/combat scenarios) to GSC class indices.
# Words not in GSC are mapped to "unknown" (label index 11).
DEMO_TO_GSC = {
    "help":    "unknown",     # not in GSC -- detected as out-of-vocab
    "water":   "unknown",
    "down":    "down",
    "here":    "unknown",
    "vehicle": "unknown",
    "drone":   "unknown",
    "shot":    "unknown",
    "yes":     "yes",
    "no":      "no",
    "up":      "up",
    "left":    "left",
    "right":   "right",
    "on":      "on",
    "off":     "off",
    "stop":    "stop",
    "go":      "go",
}


def _ssm_repo() -> Path:
    return Path(os.environ.get(
        "NC_SSM_REPO",
        REPO.parent / "NC-SSM-TASLP2026")).resolve()


def _tcn_repo() -> Path:
    return Path(os.environ.get(
        "NC_TCN_REPO",
        REPO.parent / "NC-TCN")).resolve()


class RealKWSClassifier:
    """Loads a pre-trained NC-SSM or NC-TCN classifier.

    Args:
        model: 'nc-ssm' (NanoMamba) or 'nc-tcn'
        backbone: For NC-SSM, one of {'Tiny', 'Small'}; ignored for NC-TCN.
    """

    def __init__(self, model: str = "nc-ssm", backbone: str = "Tiny"):
        self.model_name = model
        self.backbone = backbone
        self.ok = False
        self.reason = ""
        self.model = None
        self._device = "cpu"
        try:
            import torch  # noqa: F401
        except ImportError:
            self.reason = "torch not installed"
            return

        if model == "nc-ssm":
            self._load_ncssm(backbone)
        elif model == "nc-tcn":
            self._load_nctcn()
        else:
            self.reason = f"unknown model {model!r}"

    # ----------------------------------------------------------- NC-SSM
    def _load_ncssm(self, backbone: str) -> None:
        repo = _ssm_repo()
        ckpt = repo / "checkpoints_full" / f"NanoMamba-{backbone}" / "best.pt"
        if not repo.is_dir():
            self.reason = f"NC-SSM repo missing: {repo}"
            return
        if not ckpt.is_file():
            self.reason = f"checkpoint missing: {ckpt}"
            return
        sys.path.insert(0, str(repo))
        try:
            import torch
            from nanomamba import (create_nanomamba_tiny,
                                    create_nanomamba_small)
            factory = (create_nanomamba_tiny if backbone == "Tiny"
                       else create_nanomamba_small)
            net = factory()
            state = torch.load(ckpt, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            net.load_state_dict(state, strict=False)
            net.eval()
            self.model = net
            self.ok = True
            n = sum(p.numel() for p in net.parameters())
            print(f"[KWS] NC-SSM-{backbone} loaded "
                  f"({n} params, {n/1024:.1f} KB INT8)")
        except Exception as e:
            self.reason = f"load error: {e}"
            self.model = None

    # ----------------------------------------------------------- NC-TCN
    def _load_nctcn(self) -> None:
        repo = _tcn_repo()
        ckpt = repo / "checkpoints" / "nc_tcn_20k_best.pt"
        if not repo.is_dir():
            self.reason = f"NC-TCN repo missing: {repo}"
            return
        if not ckpt.is_file():
            self.reason = f"checkpoint missing: {ckpt}"
            return
        sys.path.insert(0, str(repo))
        sys.path.insert(0, str(repo / "src"))
        # NC-TCN ships its own nanomamba.py with create_nc_tcn_20k.
        # Drop any previously-cached sibling-repo nanomamba module.
        for mod in list(sys.modules):
            if mod == "nanomamba" or mod.startswith("nanomamba."):
                del sys.modules[mod]
        try:
            import importlib.util as _ilu
            ckpt_dir = repo / "src"
            spec = _ilu.spec_from_file_location(
                "nctcn_nanomamba", str(ckpt_dir / "nanomamba.py"))
            mod = _ilu.module_from_spec(spec)
            spec.loader.exec_module(mod)
            create_nc_tcn_20k = mod.create_nc_tcn_20k
            import torch
            net = create_nc_tcn_20k()
            state = torch.load(ckpt, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            net.load_state_dict(state, strict=False)
            net.eval()
            self.model = net
            self.ok = True
            n = sum(p.numel() for p in net.parameters())
            print(f"[KWS] NC-TCN-20K loaded "
                  f"({n} params, {n/1024:.1f} KB INT8)")
        except Exception as e:
            self.reason = f"load error: {e}"
            self.model = None

    # ----------------------------------------------------------- inference
    def classify(self, wav_1s: np.ndarray) -> Tuple[str, float]:
        """Run KWS on a 1-second 16 kHz mono float32 waveform.

        Returns (predicted_label, confidence).
        """
        if not self.ok or self.model is None:
            return ("unknown", 0.0)
        import torch
        x = wav_1s.astype("float32")
        if len(x) > 16000:
            x = x[:16000]
        elif len(x) < 16000:
            x = np.pad(x, (0, 16000 - len(x)))
        with torch.no_grad():
            t = torch.from_numpy(x).unsqueeze(0)
            try:
                logits = self.model(t)
            except Exception as e:
                return ("error", 0.0)
            probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()
        idx = int(np.argmax(probs))
        label = GSC_LABELS[idx] if idx < len(GSC_LABELS) else "unknown"
        return label, float(probs[idx])

    def status(self) -> str:
        if self.ok:
            return f"[OK] {self.model_name}/{self.backbone}"
        return f"[fallback] {self.model_name}/{self.backbone}: {self.reason}"


if __name__ == "__main__":
    # Quick self-test
    for model, bb in (("nc-ssm", "Tiny"), ("nc-ssm", "Small"), ("nc-tcn", "")):
        kws = RealKWSClassifier(model=model, backbone=bb or "Tiny")
        print(f"  {model:<7} {bb or '-':<7} {kws.status()}")
        if kws.ok:
            label, conf = kws.classify(np.random.randn(16000) * 0.01)
            print(f"    test classify -> {label} ({conf:.2f})")
