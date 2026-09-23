# -*- coding: utf-8 -*-
"""Figure 8: cluster-disjoint experiment (D vs L) — held-out classroom frames and clean subset."""

import os
from pathlib import Path

# Configure with environment variables, or edit the two defaults below.
#   DATA_ROOT : directory holding the datasets (must contain SHWD/VOC2028 for SHWD)
#   OUT_DIR   : directory for generated result files
DATA_ROOT = Path(os.environ.get("AUDIT_DATA_ROOT", "data"))
OUT_DIR = Path(os.environ.get("AUDIT_OUT_DIR", "results"))
OUT_DIR.mkdir(parents=True, exist_ok=True)


import json, os, io, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

AUD = r"results"
OUT = r"figures"
D_C, L_C = "#5FA8D8", "#255F9E"
INK, MUT, GRID = "#1A1A1A", "#5A5A5A", "#E4E4E1"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.linewidth": 0.7,
                     "axes.edgecolor": "#6A6A6A", "xtick.color": "#6A6A6A", "ytick.color": "#6A6A6A",
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "svg.fonttype": "none"})


def style(ax):
    ax.yaxis.grid(True, color=GRID, lw=0.7, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)


res = json.load(open(f"{AUD}/disjoint_result_multi.json"))
resg = json.load(open(f"{AUD}/disjoint_result_giant.json")) if os.path.exists(f"{AUD}/disjoint_result_giant.json") else None
panels = [("testH", "Held-out classroom frames\n(head AP@0.5)"), ("test_clean", "Clean official subset\n(mAP@0.5)")]
groups = [("multi", res)] + ([("giant", resg)] if resg else [])

fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7), width_ratios=[1.25, 1])
for ax, (sub, title) in zip(axes, panels):
    x = np.arange(len(groups)); w = 0.32
    for i, (cfg, col, lab) in enumerate((("D", D_C, "D · cluster-disjoint"), ("L", L_C, "L · with leak frames"))):
        means = [g["agg"][cfg][sub]["mAP50"]["mean"] for _, g in groups]
        stds = [g["agg"][cfg][sub]["mAP50"]["std"] for _, g in groups]
        ax.bar(x + (i - 0.5) * w, means, w, color=col, edgecolor="white", lw=0.8, zorder=3, label=lab,
               hatch="///" if cfg == "D" else None)
        ax.errorbar(x + (i - 0.5) * w, means, yerr=stds, fmt="none", ecolor=INK, elinewidth=0.8, capsize=2.5, zorder=4)
        for k, (_, g) in enumerate(groups):
            pts = [r["mAP50"] for r in g["raw"][cfg][sub]]
            ax.scatter(np.full(len(pts), x[k] + (i - 0.5) * w) + np.linspace(-0.05, 0.05, len(pts)), pts,
                       s=9, color="white", edgecolor=INK, lw=0.5, zorder=5)
    for k, (_, g) in enumerate(groups):
        d = g["agg"]["L"][sub]["mAP50"]["mean"] - g["agg"]["D"][sub]["mAP50"]["mean"]
        top = max(g["agg"][c][sub]["mAP50"]["mean"] + g["agg"][c][sub]["mAP50"]["std"] for c in "DL")
        ax.text(x[k], top + 0.35, f"L − D = {d:+.2f}".replace("-", "−"), ha="center", fontsize=7, fontweight="bold", color=INK)
    lo = min(g["agg"][c][sub]["mAP50"]["mean"] for _, g in groups for c in "DL")
    hi = max(g["agg"][c][sub]["mAP50"]["mean"] for _, g in groups for c in "DL")
    ax.set_ylim(lo - 3.0, hi + 2.2)
    ax.set_xticks(x); ax.set_xticklabels(["8 sessions held out" if n == "multi" else "largest cluster held out" for n, _ in groups], fontsize=7)
    ax.set_title(title, fontsize=8, loc="left", fontweight="bold")
    ax.set_ylabel("AP@0.5 (%)", fontsize=8); style(ax)
axes[0].legend(frameon=False, fontsize=7, loc="lower left")
fig.tight_layout(w_pad=1.2)
for ext in ("svg", "pdf"):
    fig.savefig(os.path.join(OUT, f"fig8_disjoint.{ext}"), bbox_inches="tight", pad_inches=0.06)
fig.savefig(os.path.join(OUT, "fig8_disjoint.png"), dpi=600, bbox_inches="tight", pad_inches=0.06, facecolor="white")
print("saved fig8_disjoint")
