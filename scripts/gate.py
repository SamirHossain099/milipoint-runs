"""Gate G1-G3 for the MiliPoint split audit (keypoints and identification).

Frames are taken in the official keypoint order (files 0..18 concatenated, as `_process` does); the identification
loader concatenates the files subject by subject instead (`iden_order`), and its split is rebuilt in that order. A window i holds frames
i-4..i (stacks = 5; windows cross file boundaries exactly as the official code does). The official split is rebuilt
bit for bit: `random.seed(20); random.shuffle(idx)` on the window indices, 8:1:1.

Features (no learning): per frame, centroid (3), spread (3) and point count of the radar points; 5 frames -> 35 numbers.
Keypoints: the official 9-point reduction (8 joints + mean of the 5 head points), MLE in cm (mean Euclidean error).

    python scripts/gate.py
Writes results/gate.json.
"""
import json
import os
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

ROOT = Path(__file__).resolve().parents[1]
# MILIPOINT_ROOT holds raw/ (the unzipped MiliPoint_data.zip) and code/ (a clone of github.com/yizzfz/MiliPoint)
BASE = Path(os.environ.get("MILIPOINT_ROOT", r"N:\Datasets\MiliPoint"))
RAW = BASE / "raw"
ID_JSON = BASE / "code" / "data" / "raw" / "id.json"
S = 5
KP18 = ['NOSE', 'NECK', 'RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST', 'LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_WRIST',
        'RIGHT_HIP', 'RIGHT_KNEE', 'RIGHT_ANKLE', 'LEFT_HIP', 'LEFT_KNEE', 'LEFT_ANKLE', 'RIGHT_EYE', 'LEFT_EYE',
        'RIGHT_EAR', 'LEFT_EAR']
KP9 = ['RIGHT_SHOULDER', 'RIGHT_ELBOW', 'LEFT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_HIP', 'RIGHT_KNEE', 'LEFT_HIP', 'LEFT_KNEE']
HEAD = ['NOSE', 'RIGHT_EYE', 'LEFT_EYE', 'RIGHT_EAR', 'LEFT_EAR']


def load():
    cache = ROOT / "results" / "frames.npz"           # local only (derived from the dataset)
    if cache.exists():
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    subj_of = {f: int(s) for s, fs in json.loads(ID_JSON.read_text()).items() for f in fs}
    feat, kp, fid = [], [], []
    i9, ih = [KP18.index(n) for n in KP9], [KP18.index(n) for n in HEAD]
    for f in range(19):
        for d in pickle.load(open(RAW / f"{f}.pkl", "rb")):
            x = np.asarray(d["x"], float)
            feat.append(np.r_[x.mean(0), x.std(0), len(x)] if len(x) else np.zeros(7))
            y = np.asarray(d["y"], float)
            kp.append(np.vstack([y[i9], y[ih].mean(0)]))
            fid.append(f)
    out = {"feat": np.asarray(feat, np.float32), "kp": np.asarray(kp, np.float32), "fid": np.asarray(fid, np.int16)}
    out["subj"] = np.asarray([subj_of[f] for f in out["fid"]], np.int16)
    np.savez_compressed(cache, **out)
    return out


def iden_order(fid):
    """Frame order of the released identification loader: the files of each subject in id.json order, concatenated
    (0, 10, 17, 1, 4, 11, 15, 2, 12, 3, 5, 16, 6, 7, 8, 9, 18, 13, 14). Its windows are stacked in this order."""
    files = [f for fs in json.loads(ID_JSON.read_text()).values() for f in fs]
    return np.concatenate([np.flatnonzero(fid == f) for f in files])


def windows(feat):
    """Window i = frames i-4..i, zero before the start of the whole list (as the official per_data_point padding)."""
    n = len(feat)
    W = np.zeros((n, S * feat.shape[1]), np.float32)
    for j in range(S):
        src = np.arange(n) - j
        ok = src >= 0
        W[ok, j * feat.shape[1]:(j + 1) * feat.shape[1]] = feat[src[ok]]
    return W


def official_split(n, seed=20, parts=(0.8, 0.1, 0.1)):
    idx = list(range(n))
    random.seed(seed)
    random.shuffle(idx)
    a, b = int(parts[0] * n), int(parts[0] * n) + int(parts[1] * n)
    return np.asarray(idx[:a]), np.asarray(idx[a:b]), np.asarray(idx[b:])


def mle_cm(pred, true):
    return float(np.linalg.norm(pred - true, axis=2).mean() * 100)


def nearest_in_time(te, tr_mask, n, max_k=50):
    """For each test window, the closest training window in the frame order (any distance up to max_k)."""
    out = np.full(len(te), -1)
    for k in range(1, max_k + 1):
        for sgn in (-1, 1):
            cand = te + sgn * k
            ok = (out < 0) & (cand >= 0) & (cand < n)
            ok[ok] &= tr_mask[cand[ok]]
            out[ok] = cand[ok]
        if (out >= 0).all():
            break
    return out


class MLP(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d_in, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, d_out))

    def forward(self, x):
        return self.f(x)


def train_mlp(Xtr, Ytr, Xte, task, epochs=20, seed=0):
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xt = torch.as_tensor((Xtr - mu) / sd, device=dev)
    Xv = torch.as_tensor((Xte - mu) / sd, device=dev)
    if task == "kp":
        Yt = torch.as_tensor(Ytr.reshape(len(Ytr), -1), device=dev)
        m = MLP(Xt.shape[1], Yt.shape[1]).to(dev)
        lossf = nn.MSELoss()
    else:
        Yt = torch.as_tensor(Ytr, dtype=torch.long, device=dev)
        m = MLP(Xt.shape[1], int(Ytr.max()) + 1).to(dev)
        lossf = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    g = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(len(Xt), generator=g).to(dev)
        for i in range(0, len(perm), 1024):
            j = perm[i:i + 1024]
            loss = lossf(m(Xt[j]), Yt[j])
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        out = m(Xv).cpu().numpy()
    return out.reshape(len(Xte), 9, 3) if task == "kp" else out.argmax(1)


def evaluate(W, d, tr, te, tag, res, do_time=True):
    n = len(W)
    t0 = time.time()
    r = {"n_train": int(len(tr)), "n_test": int(len(te))}
    if do_time:
        tr_mask = np.zeros(n, bool)
        tr_mask[tr] = True
        share = np.zeros(len(te), int)                       # frames shared with the closest training window
        for k in range(1, S):
            for sgn in (-1, 1):
                c = te + sgn * k
                ok = (c >= 0) & (c < n)
                hit = np.zeros(len(te), bool)
                hit[ok] = tr_mask[c[ok]]
                share = np.where(hit & (share < S - k), S - k, share)
        r["frac_test_with_train_twin_sharing_4"] = float((share == 4).mean())
        r["frac_test_sharing_any_frame"] = float((share > 0).mean())
        r["mean_frames_shared"] = float(share.mean())
        nt = nearest_in_time(te, tr_mask, n)
        okn = nt >= 0
        r["time_neighbour_kp_mle_cm"] = mle_cm(d["kp"][nt[okn]], d["kp"][te[okn]])
        r["time_neighbour_iden_acc"] = float((d["subj"][nt[okn]] == d["subj"][te[okn]]).mean())
    mu, sd = W[tr].mean(0), W[tr].std(0) + 1e-6
    Ztr, Zte = (W[tr] - mu) / sd, (W[te] - mu) / sd
    knr = KNeighborsRegressor(n_neighbors=1, n_jobs=-1).fit(Ztr, d["kp"][tr].reshape(len(tr), -1))
    r["nn1_kp_mle_cm"] = mle_cm(knr.predict(Zte).reshape(len(te), 9, 3), d["kp"][te])
    knc = KNeighborsClassifier(n_neighbors=1, n_jobs=-1).fit(Ztr, d["subj"][tr])
    r["nn1_iden_acc"] = float((knc.predict(Zte) == d["subj"][te]).mean())
    r["mlp_kp_mle_cm"] = mle_cm(train_mlp(W[tr], d["kp"][tr], W[te], "kp"), d["kp"][te])
    r["mlp_iden_acc"] = float((train_mlp(W[tr], d["subj"][tr], W[te], "iden") == d["subj"][te]).mean())
    r["seconds"] = round(time.time() - t0)
    res[tag] = r
    print(tag, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)


def main():
    d = load()
    W = windows(d["feat"])
    n = len(W)
    res = {"n_windows": n, "published": {"kp9_mle_cm": [13.12, 16.53], "iden_acc": [77.65, 95.88]}}
    tr, va, te = official_split(n)
    evaluate(W, d, tr, te, "official", res)
    # the released identification split stacks and shuffles windows in subject order, not file order
    o = iden_order(d["fid"])
    evaluate(windows(d["feat"][o]), {k: v[o] for k, v in d.items()}, tr, te, "official_iden", res)
    # keypoints, subjects held out: two folds of two single-file subjects each, and a fold of two multi-file subjects
    for held in ([9, 10], [3, 5], [0, 8]):
        m = np.isin(d["subj"], held)
        evaluate(W, d, np.flatnonzero(~m), np.flatnonzero(m), f"kp_subjects_out_{held[0]}_{held[1]}", res)
    # identification across sessions: multi-file subjects, the last file of each held out
    multi = {0: [0, 10, 17], 1: [1, 4, 11, 15], 2: [2, 12], 4: [5, 16], 8: [9, 18]}
    held_files = [fs[-1] for fs in multi.values()]
    keep = np.isin(d["subj"], list(multi))
    te_s = np.flatnonzero(keep & np.isin(d["fid"], held_files))
    tr_s = np.flatnonzero(keep & ~np.isin(d["fid"], held_files))
    evaluate(W, d, tr_s, te_s, "iden_sessions_out", res)
    # the same five subjects under the official-style random split, for a like-for-like identification comparison
    sub = np.flatnonzero(keep)
    rng = np.random.default_rng(20)
    p = rng.permutation(sub)
    evaluate(W, d, np.sort(p[:int(0.8 * len(p))]), np.sort(p[int(0.9 * len(p)):]), "iden_random_same5", res)
    (ROOT / "results" / "gate.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
