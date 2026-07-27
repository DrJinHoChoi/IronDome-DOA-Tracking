# -*- coding: utf-8 -*-
"""
MC 결과 표(eval_mc.py 출력 .txt)를 파싱해 논문용 focused 그림 생성.

생성:
  fig_loop_2x2.png  : k6stat — 2x2 (front-end x loop) locRMSE 막대 (Theorem 2 일반성)
  fig_gate.png      : k10 — 게이트가 이동 시 폐루프 악화를 막음 (Pd, locRMSE)

사용: python revision/make_figs.py
입력: revision/mc_k6stat_200.txt, revision/mc_k10_200.txt (없으면 _500 도 시도)
"""
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PIPES = ["COP-RFS", "COP-RFS-CL", "COP-RFS-GCL", "MUSIC-PHD", "MUSIC-PHD-CL"]
FLOAT = re.compile(r"[-+]?\d+\.\d+|\d+")


def _read_text(path):
    for enc in ("utf-16", "utf-8", "cp949", "latin-1"):
        try:
            with open(path, "r", encoding=enc) as f:
                t = f.read()
            if "Pipeline" in t or "COP" in t:
                return t
        except Exception:
            continue
    with open(path, "rb") as f:
        return f.read().decode("latin-1", "ignore")


def parse(path):
    """{pipeline: {pd,pd_e,loc,loc_e,gospa,gospa_e,gloc,gmiss,gfa,sw,sw_e}}."""
    if not os.path.exists(path):
        return {}
    out = {}
    for line in _read_text(path).splitlines():
        toks = line.strip().split(None, 1)
        if len(toks) != 2 or toks[0] not in PIPES:
            continue
        nums = [float(x) for x in FLOAT.findall(toks[1])]
        if len(nums) < 11:
            continue
        k = ["pd", "pd_e", "loc", "loc_e", "gospa", "gospa_e",
             "gloc", "gmiss", "gfa", "sw", "sw_e"]
        out[toks[0]] = dict(zip(k, nums[:11]))
    return out


def _pick(stem):
    for suf in ("_200", "_500", "_baseline"):
        p = os.path.join(HERE, f"mc_{stem}{suf}.txt")
        if os.path.exists(p):
            return p
    return os.path.join(HERE, f"mc_{stem}_200.txt")


def fig_2x2(d):
    """k6stat: locRMSE for COP/MUSIC x open/closed."""
    groups = [("COP", "COP-RFS", "COP-RFS-CL"),
              ("MUSIC", "MUSIC-PHD", "MUSIC-PHD-CL")]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    x = np.arange(len(groups)); w = 0.36
    op = [d.get(g[1], {}).get("loc", np.nan) for g in groups]
    op_e = [d.get(g[1], {}).get("loc_e", 0) for g in groups]
    cl = [d.get(g[2], {}).get("loc", np.nan) for g in groups]
    cl_e = [d.get(g[2], {}).get("loc_e", 0) for g in groups]
    ax.bar(x - w/2, op, w, yerr=op_e, capsize=4, label="open-loop", color="#888")
    ax.bar(x + w/2, cl, w, yerr=cl_e, capsize=4, label="closed-loop", color="#1f77b4")
    ax.set_xticks(x); ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel("localization RMSE (deg)")
    ax.set_title("Closed loop improves BOTH front-ends\n(stationary, SNR=5 dB)")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    for xi, (o, c) in enumerate(zip(op, cl)):
        if np.isfinite(o) and np.isfinite(c) and o > 0:
            ax.text(xi, max(o, c) + 0.03, f"-{100*(o-c)/o:.0f}%",
                    ha="center", fontsize=9, color="#1f77b4")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_loop_2x2.png"), dpi=150)
    plt.close(fig)
    print("saved fig_loop_2x2.png")


def fig_gate(d):
    """k10: Pd and locRMSE for open / ungated-CL / gated-GCL / MUSIC."""
    order = ["COP-RFS", "COP-RFS-CL", "COP-RFS-GCL", "MUSIC-PHD"]
    labels = ["COP\n(open)", "COP-CL\n(ungated)", "COP-GCL\n(gated)", "MUSIC"]
    pd = [d.get(p, {}).get("pd", np.nan) for p in order]
    pd_e = [d.get(p, {}).get("pd_e", 0) for p in order]
    loc = [d.get(p, {}).get("loc", np.nan) for p in order]
    loc_e = [d.get(p, {}).get("loc_e", 0) for p in order]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
    cols = ["#2ca02c", "#d62728", "#1f77b4", "#888"]
    a1.bar(range(4), pd, yerr=pd_e, capsize=4, color=cols)
    a1.set_xticks(range(4)); a1.set_xticklabels(labels, fontsize=9)
    a1.set_ylabel("detection rate (%)"); a1.set_title("(a) Detection (K=10 > M-1=7)")
    a1.grid(True, axis="y", alpha=0.3)
    a2.bar(range(4), loc, yerr=loc_e, capsize=4, color=cols)
    a2.set_xticks(range(4)); a2.set_xticklabels(labels, fontsize=9)
    a2.set_ylabel("localization RMSE (deg)")
    a2.set_title("(b) Ungated CL degrades; gate restores open-loop")
    a2.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_gate.png"), dpi=150)
    plt.close(fig)
    print("saved fig_gate.png")


if __name__ == "__main__":
    d6 = parse(_pick("k6stat"))
    d10 = parse(_pick("k10"))
    print("k6stat pipelines:", list(d6))
    print("k10 pipelines:", list(d10))
    if d6:
        fig_2x2(d6)
    if d10:
        fig_gate(d10)
