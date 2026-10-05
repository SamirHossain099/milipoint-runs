"""Figure 1: identification accuracy by split: mean over seeds (marker) and range (bar), chance marked. Reads results/ only.

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
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 8, "axes.edgecolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})


def j(n):
    return json.loads((R / n).read_text())


def main():
    pn = j("pointnet_iden.json")
    pm = [j("pointmlp_iden.json"), j("pointmlp_iden_s1.json")]

    def both(k):
        return [v for p in pm for v in p[k]]
    rows = [("Released split\n(random windows, 11 participants)", both("official"), pn["official"], 1 / 11),
            ("Random windows\n(5 participants)", both("random_same5"), pn["random_same5"], 0.2),
            ("Random windows, smaller\ntraining set (5 participants)", both("random_same5_sizematched"),
             pn["random_same5_sizematched"], 0.2),
            ("Held-out recording run\n(same 5 participants)", both("sessions_out"), pn["sessions_out"], 0.2)]
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for i, (_, a, b, ch) in enumerate(rows):
        for vals, y, mk, ms, col, lab in ((a, i - 0.14, "o", 5.5, SLOT[0], "PointMLP (authors' model)"),
                                          (b, i + 0.14, "s", 5, SLOT[1], "PointNet")):
            if not vals:
                continue
            v = 100 * np.asarray(vals)
            ax.plot([v.min(), v.max()], [y, y], color=col, lw=1.5, solid_capstyle="round")   # range over seeds
            ax.plot([v.mean()], [y], mk, ms=ms, color=col, mec="white", mew=0.8, ls="none",
                    label=lab if i == 0 else None)
        ax.plot([100 * ch] * 2, [i - 0.3, i + 0.3], color=INK2, lw=1.2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Identification accuracy (%)", color=INK)
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.text(100 / 11 + 1.5, -0.36, "chance", fontsize=7, color=INK2, va="center")
    ax.legend(frameon=False, fontsize=7, loc="lower center", bbox_to_anchor=(0.35, 1.0), ncol=2, labelcolor=INK)
    fig.tight_layout()
    (ROOT / "figures").mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "figures" / f"fig1.{ext}", dpi=600, facecolor="white")
    print("wrote figures/fig1.png, fig1.pdf")


if __name__ == "__main__":
    main()
