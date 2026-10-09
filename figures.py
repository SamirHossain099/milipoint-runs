"""Figure 1: identification accuracy on each participant's held-out run, per model: mean over seeds (marker) and range
(bar), the released-settings PointMLP run where it exists (hollow marker), each model's accuracy on random windows
with the same training size (dashed), and chance. Reads results/ only.

    python figures.py
Writes figures/fig1.png (600 dpi) and figures/fig1.pdf.
"""
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent
R = ROOT / "results"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
SLOT = ["#2a78d6", "#eb6834"]                      # reference palette slots 1-2, validated all-pairs
PARTICIPANTS = [0, 1, 2, 4, 8]                     # label k of the five-participant runs is PARTICIPANTS[k]
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 8, "axes.edgecolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})


def j(n):
    f = R / n
    return json.loads(f.read_text()) if f.exists() else None


def runs(files):
    """Per-run accuracy per seed, and the size-matched random-window accuracy per seed, pooled over result files."""
    per, sm = [], []
    for r in filter(None, (j(f) for f in files)):
        per += r["sessions_out_per_subject"]
        sm += r["random_same5_sizematched"]
    return np.array([[p[str(k)] for k in range(5)] for p in per]) * 100, np.array(sm) * 100


def main():
    models = [("PointMLP (authors' model)", ["pointmlp_iden.json", "pointmlp_iden_s1.json"], SLOT[0], "o", -0.13),
              ("PointNet", ["pointnet_iden.json"], SLOT[1], "s", 0.13)]
    rel = j("pointmlp_released_s0_sessions_out.json")
    rel_sm = j("pointmlp_released_s0_random_same5_sizematched.json")
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for name, files, col, mk, dx in models:
        per, sm = runs(files)
        x = np.arange(5) + dx
        ax.vlines(x, per.min(0), per.max(0), color=col, lw=1.5)                   # range over seeds
        ax.plot(x, per.mean(0), mk, ms=5, color=col, mec="white", mew=0.8, ls="none", label=name)
        ax.axhline(sm.mean(), color=col, lw=0.9, ls=(0, (4, 2)))                  # random windows, same training size
    if rel:
        ax.plot(np.arange(5) - 0.13, [100 * rel["per_subject"][str(k)] for k in range(5)], "o", ms=5, mfc="none",
                mec=SLOT[0], mew=1.0, ls="none", label="PointMLP, released settings")
        if rel_sm:
            ax.axhline(100 * rel_sm["acc"], color=SLOT[0], lw=0.9, ls=(0, (1, 1.5)))
    ax.axhline(20, color=INK2, lw=1.0)
    ax.text(4.45, 17.5, "chance", fontsize=7, color=INK2, ha="right", va="top")
    ax.text(-0.45, 88.5, "dashed, dotted: random windows, same training size", fontsize=7, color=INK2, ha="left",
            va="top")
    ax.set_xticks(range(5))
    ax.set_xticklabels([f"P{p}" for p in PARTICIPANTS], color=INK)
    ax.set_xlabel("Participant whose last run is held out", color=INK)
    ax.set_ylabel("Accuracy on the held-out run (%)", color=INK)
    ax.set_ylim(-3, 100)
    ax.set_xlim(-0.5, 4.5)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.legend(frameon=False, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, labelcolor=INK)
    fig.tight_layout()
    (ROOT / "figures").mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "figures" / f"fig1.{ext}", dpi=600, facecolor="white")
    print("wrote figures/fig1.png, fig1.pdf")


if __name__ == "__main__":
    main()
