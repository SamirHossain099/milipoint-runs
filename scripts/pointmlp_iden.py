"""Identification with the authors' own PointMLP (MiliPoint repository, mmrnet/models/pointmlp.py, unchanged).

The model imports `fps` and `knn` from torch_geometric.nn, which need compiled extensions with no build for this
PyTorch. A two-function stand-in is placed in sys.modules before the authors' file is loaded:
  fps(x, batch, ratio)          farthest-point sampling per sample (equal-size samples) from a random start point,
                                 as torch_cluster's default, global indices into x
  knn(x, y, k, batch_x, batch_y) the k nearest x for each y within the same sample; returns (row, col), col global
Nothing else of torch_geometric is used by PointMLP.

Input: the stacked window flattened to 110 points (5 frames x 22, official sampling). Splits as pointnet_iden.py.

    python scripts/pointmlp_iden.py
Writes results/pointmlp_iden.json.
"""
import importlib.util
import json
import sys
import time
import types
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate import BASE, S, iden_order, load, official_split  # noqa: E402
from pointnet_iden import K, frames, score, splits  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CODE = BASE / "code" / "mmrnet" / "models" / "pointmlp.py"


def fps(x, batch, ratio):
    B = int(batch.max()) + 1
    N = x.shape[0] // B
    s = int(round(ratio * N))
    X = x.reshape(B, N, -1)
    idx = torch.zeros(B, s, dtype=torch.long, device=x.device)
    dist = torch.full((B, N), float("inf"), device=x.device)
    far = torch.randint(N, (B,), device=x.device)                       # random start, as torch_cluster
    ar = torch.arange(B, device=x.device)
    for i in range(s):
        idx[:, i] = far
        d = ((X - X[ar, far][:, None, :]) ** 2).sum(-1)
        dist = torch.minimum(dist, d)
        far = dist.argmax(1)
    return (idx + ar[:, None] * N).reshape(-1)


def knn(x, y, k, batch_x, batch_y):
    B = int(batch_x.max()) + 1
    N, M = x.shape[0] // B, y.shape[0] // B
    X, Y = x.reshape(B, N, -1), y.reshape(B, M, -1)
    nn_idx = torch.cdist(Y, X).topk(k, dim=2, largest=False).indices          # B x M x k
    col = (nn_idx + torch.arange(B, device=x.device)[:, None, None] * N).reshape(-1)
    row = (torch.arange(B * M, device=x.device)[:, None].expand(-1, k)).reshape(-1)
    return row, col


def load_pointmlp():
    tg = types.ModuleType("torch_geometric")
    tgnn = types.ModuleType("torch_geometric.nn")
    tgnn.fps, tgnn.knn = fps, knn
    tg.nn = tgnn
    sys.modules["torch_geometric"], sys.modules["torch_geometric.nn"] = tg, tgnn
    spec = importlib.util.spec_from_file_location("pointmlp_authors", CODE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PointMLP


def run(PointMLP, P, labels, tr, te, n_cls, seed, epochs=3, dev="cuda", per_class=False):
    torch.manual_seed(seed)
    Pt = torch.as_tensor(P, device=dev)
    Y = torch.as_tensor(labels, dtype=torch.long, device=dev)
    offs = torch.arange(S - 1, -1, -1, device=dev)

    def batch(idx):
        src = idx[:, None] - offs[None, :]
        w = Pt[src.clamp(min=0)]
        w[src < 0] = 0
        return w.reshape(len(idx), S * K, 3)

    m = PointMLP({"num_classes": n_cls, "max_points": K, "stacks": S}).to(dev)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    steps = epochs * ((len(tr) + 255) // 256)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    trt, tet = torch.as_tensor(tr, device=dev), torch.as_tensor(te, device=dev)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        m.train()
        perm = trt[torch.randperm(len(trt), generator=g).to(dev)]
        for i in range(0, len(perm) - 255, 256):
            j = perm[i:i + 256]
            loss = nn.functional.cross_entropy(m(batch(j)), Y[j])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    m.eval()
    with torch.no_grad():
        pred = torch.cat([m(batch(tet[i:i + 2048])).argmax(1) for i in range(0, len(tet), 2048)]).cpu().numpy()
    return score(pred, labels[te], per_class)


def main(seeds=(0,)):
    PointMLP = load_pointmlp()
    d = load()
    P = frames()
    n = len(P)
    res = {"published_pointmlp_iden": 95.88}
    t0 = time.time()
    tr, _, te = official_split(n)
    o = iden_order(d["fid"])                                   # the released identification loader's frame order
    res["official"] = [run(PointMLP, P[o], d["subj"][o], tr, te, 11, s) for s in seeds]
    print("official", res["official"], f"{time.time() - t0:.0f}s", flush=True)
    for k in ("sessions_out", "random_same5", "random_same5_sizematched", "sessions_out_per_subject"):
        res[k] = []
    for s in seeds:
        y5, sp = splits(d, s)
        for k, (a, b) in sp.items():
            r = run(PointMLP, P, y5, a, b, 5, s, per_class=k == "sessions_out")
            if k == "sessions_out":
                r, pc = r
                res["sessions_out_per_subject"].append(pc)
            res[k].append(r)
        print("seed", s, {k: res[k][-1] for k in sp}, f"{time.time() - t0:.0f}s", flush=True)
    name = "pointmlp_iden.json" if tuple(seeds) == (0,) else f"pointmlp_iden_s{'-'.join(map(str, seeds))}.json"
    (ROOT / "results" / name).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(tuple(int(s) for s in sys.argv[1:]) or (0,))
