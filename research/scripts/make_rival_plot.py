"""Render the measured head-to-head figure from research/results/rival_benchmark.json.

  USE_TF=0 python3 research/scripts/make_rival_plot.py [--json PATH] [--out PATH]

Separate from make_plots.py (which is coupled to the 51-language + T4 JSONs). Writes
assets/veya_vs_rival.png using the validated brand palette.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
ASSETS = os.path.join(REPO, "assets")
DEFAULT_JSON = os.path.join(REPO, "research", "results", "rival_benchmark.json")

BLUE, ORANGE = "#2a78d6", "#eb6834"
GREEN, INK3 = "#3aa76d", "#8a8984"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLORS = [BLUE, ORANGE, GREEN, INK3, "#8e6bd1", "#c9a227"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "text.color": INK,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.linewidth": 1.0,
    "xtick.major.size": 0, "ytick.major.size": 0,
})


def strip(ax, keep=("bottom",)):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=DEFAULT_JSON)
    ap.add_argument("--out", default=os.path.join(ASSETS, "veya_vs_rival.png"))
    a = ap.parse_args()
    data = json.load(open(a.json))
    suites = data["suites"]
    names = sorted({m for blk in suites.values() for m in blk if "accuracy" in blk[m]})
    if not names:
        raise SystemExit("no measured models in %s" % a.json)
    suite_names = sorted(suites)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(16, 6.5), gridspec_kw={"width_ratios": [2, 1]})
    x = np.arange(len(suite_names)); w = 0.8 / max(1, len(names))
    for i, nm in enumerate(names):
        acc = [suites[s].get(nm, {}).get("accuracy", np.nan) for s in suite_names]
        ece = [suites[s].get(nm, {}).get("ece", np.nan) for s in suite_names]
        off = (i - (len(names) - 1) / 2) * w
        axA.bar(x + off, acc, w, color=COLORS[i % len(COLORS)], label=nm, zorder=3)
        axB.bar(x + off, ece, w, color=COLORS[i % len(COLORS)], label=nm, zorder=3)
    axA.set_xticks(x); axA.set_xticklabels(suite_names, rotation=35, ha="right", fontsize=8)
    axA.set_ylabel("accuracy"); axA.set_ylim(0, 1.05)
    axA.set_title("Measured head-to-head: accuracy", fontsize=13, fontweight="bold", loc="left")
    axA.yaxis.grid(True, color=GRID, lw=0.9); axA.set_axisbelow(True); strip(axA)
    axA.legend(frameon=False, fontsize=8.5, loc="upper right", labelcolor=INK2)
    axB.set_xticks(x); axB.set_xticklabels(suite_names, rotation=35, ha="right", fontsize=8)
    axB.set_ylabel("ECE (lower is better)")
    axB.set_title("Calibration", fontsize=13, fontweight="bold", loc="left")
    axB.yaxis.grid(True, color=GRID, lw=0.9); axB.set_axisbelow(True); strip(axB)
    fig.suptitle("Veya vs rivals  ·  identical questions, one machine  ·  %s"
                 % data["meta"].get("timestamp", ""), fontsize=14, fontweight="bold",
                 color=INK, x=0.012, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=150, bbox_inches="tight")
    print("wrote %s (%.0f KB)" % (a.out, os.path.getsize(a.out) / 1024))


if __name__ == "__main__":
    main()
