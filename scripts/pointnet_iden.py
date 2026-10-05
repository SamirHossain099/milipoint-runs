"""Identification with a point network on the raw radar points, official split vs held-out recording sessions.

Points per frame follow the official sampling (zero-pad to 22, then a random choice of 22 without replacement); a window
is frames i-4..i (stacks 5, as the released configs). The network is PointNet-style: a shared point MLP, max-pooled
within each frame, the five frame codes concatenated and classified. Settings, 3 seeds each:
  official      all 11 subjects, the released split rebuilt: frames in the identification loader's subject order,
                windows i-4..i in that order, shuffled with seed 20, 8:1:1
  random_same5  the five subjects with several recording files, random window split 8:1:1
  random_same5_sizematched  the same, training set cut to the size of the run-disjoint training set
  sessions_out  the same five subjects, the last recording file of each held out for testing (with per-subject accuracy)

    python scripts/pointnet_iden.py
Writes results/pointnet_iden.json.
"""
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate import RAW, S, iden_order, load, official_split  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
K = 22
MULTI = {0: [0, 10, 17], 1: [1, 4, 11, 15], 2: [2, 12], 4: [5, 16], 8: [9, 18]}


def frames():
    cache = ROOT / "results" / "points22.npy"           # local only (derived from the dataset)
    if cache.exists():
        return np.load(cache)
    rng = np.random.default_rng(0)
    out = []
    for f in range(19):
        for d in pickle.load(open(RAW / f"{f}.pkl", "rb")):
            x = np.asarray(d["x"], np.float32)
            x = np.pad(x, ((0, max(K - len(x), 0)), (0, 0)))
            out.append(x[rng.choice(len(x), K, replace=False)])
    P = np.stack(out)
    np.save(cache, P)
    return P


class PointNetWin(nn.Module):
    def __init__(self, n_cls):
        super().__init__()
        self.point = nn.Sequential(nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
                                   nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
                                   nn.Conv1d(128, 256, 1), nn.BatchNorm1d(256), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(S * 256, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, n_cls))

    def forward(self, w):                                 # w: B x S x K x 3
        b = w.shape[0]
        z = self.point(w.reshape(b * S, K, 3).transpose(1, 2)).max(2).values   # (B*S) x 256
        return self.head(z.reshape(b, S * 256))


def run(P, labels, tr, te, n_cls, seed, epochs=8, dev="cuda", per_class=False):
    torch.manual_seed(seed)
    Pt = torch.as_tensor(P, device=dev)
    Y = torch.as_tensor(labels, dtype=torch.long, device=dev)
    offs = torch.arange(S - 1, -1, -1, device=dev)                       # frames i-4 .. i

    def batch(idx):
        src = idx[:, None] - offs[None, :]
        w = Pt[src.clamp(min=0)]
        w[src < 0] = 0                                                     # zero before the start, as the official code
        return w

    m = PointNetWin(n_cls).to(dev)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    trt, tet = torch.as_tensor(tr, device=dev), torch.as_tensor(te, device=dev)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        m.train()
        perm = trt[torch.randperm(len(trt), generator=g).to(dev)]
        for i in range(0, len(perm), 512):
            j = perm[i:i + 512]
            loss = nn.functional.cross_entropy(m(batch(j)), Y[j])
            opt.zero_grad()
            loss.backward()
            opt.step()
    m.eval()
    with torch.no_grad():
        pred = torch.cat([m(batch(tet[i:i + 4096])).argmax(1) for i in range(0, len(tet), 4096)]).cpu().numpy()
    return score(pred, labels[te], per_class)


def score(pred, true, per_class=False):
    acc = float((pred == true).mean())
    if not per_class:
        return acc
    return acc, {str(c): float((pred[true == c] == c).mean()) for c in np.unique(true)}


def splits(d, s):
    """The five multi-file subjects (labels 0-4): run-disjoint, random, and random with the run-disjoint train size."""
    keep = np.isin(d["subj"], list(MULTI))
    remap = {k: i for i, k in enumerate(sorted(MULTI))}
    y5 = np.array([remap.get(int(x), -1) for x in d["subj"]])
    held = [fs[-1] for fs in MULTI.values()]
    te_s = np.flatnonzero(keep & np.isin(d["fid"], held))
    tr_s = np.flatnonzero(keep & ~np.isin(d["fid"], held))
    p = np.random.default_rng(20 + s).permutation(np.flatnonzero(keep))
    tr_r, te_r = p[:int(0.8 * len(p))], p[int(0.9 * len(p)):]
    return y5, {"sessions_out": (tr_s, te_s), "random_same5": (tr_r, te_r),
                "random_same5_sizematched": (tr_r[:len(tr_s)], te_r)}


def main(seeds=(0, 1, 2)):
    d = load()
    P = frames()
    n = len(P)
    res = {}
    tr, _, te = official_split(n)
    o = iden_order(d["fid"])                                   # the released identification loader's frame order
    res["official"] = [run(P[o], d["subj"][o], tr, te, 11, s) for s in seeds]
    print("official", res["official"], flush=True)

    for k in ("sessions_out", "random_same5", "random_same5_sizematched", "sessions_out_per_subject"):
        res[k] = []
    for s in seeds:
        y5, sp = splits(d, s)
        for k, (a, b) in sp.items():
            r = run(P, y5, a, b, 5, s, per_class=k == "sessions_out")
            if k == "sessions_out":
                r, pc = r
                res["sessions_out_per_subject"].append(pc)
            res[k].append(r)
        print("seed", s, {k: res[k][-1] for k in sp}, flush=True)
    res["chance"] = {"official": 1 / 11, "five_subjects": 0.2}
    (ROOT / "results" / "pointnet_iden.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    random.seed(0)
    main()
