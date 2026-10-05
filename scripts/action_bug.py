"""How much does the action stacking bug matter? The released MMRActionData filters unlabelled frames, then checks
`self.action_label[i] == self.action_label[i - j]` with i, j positions in the FILTERED list, against the UNFILTERED
label array. This counts, over all windows, the frame slots whose inclusion differs from the correct test
(`act[i] == act[i - j]` on the filtered labels), for the released configs (stacks = 5) and the paper's 50 frames.

    python scripts/action_bug.py
Writes results/action_bug.json.
"""
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def count(a, stacks):
    act = a[a != -1]
    n = len(act)
    i = np.arange(n)
    differ_slots = 0
    windows_affected = np.zeros(n, bool)
    wrong_label_slots = 0
    windows_foreign = np.zeros(n, bool)
    for j in range(1, stacks):
        ok = i - j >= 0
        bug = np.zeros(n, bool)
        bug[ok] = a[i[ok]] == a[i[ok] - j]                     # released: unfiltered array, filtered positions
        right = np.zeros(n, bool)
        right[ok] = act[i[ok]] == act[i[ok] - j]
        d = bug != right
        differ_slots += int(d.sum())
        windows_affected |= d
        wrong_label_slots += int((bug & ~right).sum())          # frames of another action stacked into the window
        windows_foreign |= bug & ~right
    return {"n_windows": int(n), "slots_total": int(n * (stacks - 1)), "slots_differ": differ_slots,
            "frac_slots_differ": differ_slots / (n * (stacks - 1)), "frac_windows_affected": float(windows_affected.mean()),
            "slots_with_foreign_action": wrong_label_slots,
            "frac_slots_foreign": wrong_label_slots / (n * (stacks - 1)),
            "frac_windows_with_foreign_action": float(windows_foreign.mean())}


def main():
    a = np.load(Path(os.environ.get("MILIPOINT_ROOT", r"N:\Datasets\MiliPoint")) / "raw" / "action_label.npy")
    res = {f"stacks_{s}": count(a, s) for s in (5, 50)}
    print(json.dumps(res, indent=1))
    (ROOT / "results" / "action_bug.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
