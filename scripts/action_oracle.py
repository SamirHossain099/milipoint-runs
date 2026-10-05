"""Action task without learning: what a split lets a model get by copying neighbours.

Labelled frames only (the official filter), in file order. Windows, each described by the mean and std over its
frames of the per-frame centroid, spread and point count (14 numbers):
  trailing   the frame and up to 49 previous labelled frames, whatever their action (label of the last frame); no
             label is used to build a window
  makgcn     the stacking of the MAK-GCN code (github.com/Gbouna/MAK-GCN, MiliPoint/data.py): frames cut into runs of
             one action, 50-frame windows every 40 frames within a run, a short window padded by repeating its last
             frame. Window boundaries follow the labels, as in that code.
Splits: contiguous (8:1:1 in file order, test = last 10%, as the released action loader) and shuffled (windows
shuffled before 8:1:1: seed 20 as the released configs for trailing, seed 42 as the MAK-GCN code for makgcn).
Per split: for makgcn, the share of test windows whose run of one action also gave a training window;
time-neighbour (the action of the nearest training window in window order; no input used) and input 1-NN.
Also records which actions each recording file holds.

    python scripts/action_oracle.py
Writes results/action_oracle.json.
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate import RAW, load, nearest_in_time  # noqa: E402
from pointnet_iden import MULTI  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
S50, STRIDE = 50, 40


def trailing_features(F):
    """Mean and std over frames max(0, i-49)..i of the labelled list."""
    n = len(F)
    cs = np.cumsum(np.vstack([np.zeros((1, F.shape[1])), F]), 0)
    cs2 = np.cumsum(np.vstack([np.zeros((1, F.shape[1])), F ** 2]), 0)
    start = np.maximum(np.arange(n) - S50 + 1, 0)
    cnt = (np.arange(n) - start + 1)[:, None]
    mean = (cs[np.arange(n) + 1] - cs[start]) / cnt
    var = (cs2[np.arange(n) + 1] - cs2[start]) / cnt - mean ** 2
    return np.hstack([mean, np.sqrt(np.clip(var, 0, None))]).astype(np.float32)


def makgcn_windows(F, act):
    """Windows as MAK-GCN's stack_and_padd_frames: per run of one label, starts 0, 40, 80, ...; pad with the last frame."""
    X, y, run = [], [], []
    cut = np.flatnonzero(np.diff(act)) + 1
    for r, (a, b) in enumerate(zip(np.r_[0, cut], np.r_[cut, len(act)])):
        for i in range(a, b, STRIDE):
            idx = np.arange(i, min(i + S50, b))
            idx = np.r_[idx, np.full(S50 - len(idx), idx[-1])]
            X.append(np.r_[F[idx].mean(0), F[idx].std(0)])
            y.append(act[a])
            run.append(r)
    return np.asarray(X, np.float32), np.asarray(y), np.asarray(run)


def same_run_in_train(run, tr, te):
    """Fraction of test windows with a training window cut from the same run of one action (one performance)."""
    return float(np.isin(run[te], run[tr]).mean())


def evaluate(X, act, tr, te):
    n = len(act)
    mask = np.zeros(n, bool)
    mask[tr] = True
    nt = nearest_in_time(te, mask, n, max_k=5000)
    ok = nt >= 0
    tn = float((act[nt[ok]] == act[te[ok]]).mean()) if ok.any() else None   # None: no training window within 5,000
    #   positions (the contiguous test set is one block)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    knn = KNeighborsClassifier(n_neighbors=1, n_jobs=-1).fit((X[tr] - mu) / sd, act[tr])
    nn1 = float((knn.predict((X[te] - mu) / sd) == act[te]).mean())
    return {"time_neighbour_acc": tn, "input_1nn_acc": nn1, "n_train": int(len(tr)), "n_test": int(len(te))}


def split(n, shuffle_seed=None):
    idx = list(range(n))
    if shuffle_seed is not None:
        random.seed(shuffle_seed)
        random.shuffle(idx)
    idx = np.asarray(idx)
    a, b = int(0.8 * n), int(0.8 * n) + int(0.1 * n)
    return idx[:a], idx[b:]


def coverage(d, a):
    lab = a != -1
    per_file = {str(f): sorted(int(x) for x in np.unique(a[(d["fid"] == f) & lab])) for f in range(19)}
    held = {}
    for s, fs in MULTI.items():
        seen = set().union(*(per_file[str(f)] for f in fs[:-1]))
        held[str(s)] = {"file": fs[-1], "actions": per_file[str(fs[-1])],
                        "not_in_training_runs": sorted(set(per_file[str(fs[-1])]) - seen)}
    return {"frac_frames_labelled": float(lab.mean()), "n_frames": int(len(a)), "actions_per_file": per_file,
            "n_actions_per_file": {f: len(v) for f, v in per_file.items()}, "held_out_run_actions": held}


def main():
    d = load()
    a = np.load(RAW / "action_label.npy")
    lab = np.flatnonzero(a != -1)
    act = a[lab]
    F = d["feat"][lab]
    res = {"published": {"pointnet++_top1": 34.45, "makgcn_top1": 98.25, "tunau_2025_top1": 98.57}, "chance": 1 / 49,
           "coverage": coverage(d, a)}
    Xt = trailing_features(F)
    res["trailing"] = {"contiguous": evaluate(Xt, act, *split(len(act))),
                       "shuffled": evaluate(Xt, act, *split(len(act), 20))}
    Xm, ym, run = makgcn_windows(F, act)
    res["makgcn"] = {"n_windows": int(len(ym)), "n_runs": int(run.max() + 1)}
    for k, sd in (("contiguous", None), ("shuffled", 42)):
        tr, te = split(len(ym), sd)
        res["makgcn"][k] = evaluate(Xm, ym, tr, te) | {"frac_test_same_run_in_train": same_run_in_train(run, tr, te)}
    for w in ("trailing", "makgcn"):
        for k in ("contiguous", "shuffled"):
            r = res[w][k]
            print(w, k, json.dumps({x: round(v, 4) if isinstance(v, float) else v for x, v in r.items()}), flush=True)
    (ROOT / "results" / "action_oracle.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
