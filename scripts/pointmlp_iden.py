"""Identification with the authors' own PointMLP (MiliPoint repository, mmrnet/models/pointmlp.py, unchanged).

The model imports `fps` and `knn` from torch_geometric.nn, which need compiled extensions with no build for this
PyTorch. A two-function stand-in is placed in sys.modules before the authors' file is loaded:
  fps(x, batch, ratio)          farthest-point sampling per sample (equal-size samples) from a random start point,
                                 as torch_cluster's default, global indices into x
  knn(x, y, k, batch_x, batch_y) the k nearest x for each y within the same sample; returns (row, col), col global
Nothing else of torch_geometric is used by PointMLP.

Input: the stacked window flattened to 110 points (5 frames x 22, official sampling). Splits as pointnet_iden.py.

    python scripts/pointmlp_iden.py [seed ...]        (3 epochs, lr 1e-3, batch 256: the main runs)
    python scripts/pointmlp_iden.py --released 0 sessions_out              (the released 100-epoch settings)
    python scripts/pointmlp_iden.py --released 0 random_same5_sizematched
Writes results/pointmlp_iden*.json and results/pointmlp_released_s<seed>_<split>.json.
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


FAST = {"epochs": 3, "lr": 1e-3, "wd": 0.0, "bs": 256, "eta_min": 0.0, "per_step": True, "drop_last": True}
# the released defaults (mmrnet/cli.py, session/wrapper.py): Adam, lr 1e-5, weight decay 1e-5, cosine decay to 1e-6 over
# the epochs, batch 128, 100 epochs
RELEASED = {"epochs": 100, "lr": 1e-5, "wd": 1e-5, "bs": 128, "eta_min": 1e-6, "per_step": False, "drop_last": False}


def run(PointMLP, P, labels, tr, te, n_cls, seed, dev="cuda", per_class=False, cfg=FAST, report_at=(), ckpt=None):
    torch.manual_seed(seed)
    Pt = torch.as_tensor(P, device=dev)
    Y = torch.as_tensor(labels, dtype=torch.long, device=dev)
    offs = torch.arange(S - 1, -1, -1, device=dev)

    def batch(idx):
        src = idx[:, None] - offs[None, :]
        w = Pt[src.clamp(min=0)]
        w[src < 0] = 0
        return w.reshape(len(idx), S * K, 3)

    def evaluate():
        m.eval()
        with torch.no_grad():
            pred = torch.cat([m(batch(tet[i:i + 2048])).argmax(1) for i in range(0, len(tet), 2048)]).cpu().numpy()
        return score(pred, labels[te], per_class)

    m = PointMLP({"num_classes": n_cls, "max_points": K, "stacks": S}).to(dev)
    opt = torch.optim.Adam(m.parameters(), cfg["lr"], weight_decay=cfg["wd"])
    bs = cfg["bs"]
    per_epoch = (len(tr) // bs) if cfg["drop_last"] else -(-len(tr) // bs)
    t_max = cfg["epochs"] * -(-len(tr) // bs) if cfg["per_step"] else cfg["epochs"]   # as the original per-step schedule
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, t_max, eta_min=cfg["eta_min"])
    trt, tet = torch.as_tensor(tr, device=dev), torch.as_tensor(te, device=dev)
    g = torch.Generator().manual_seed(seed)
    trace, start = {}, 1
    if ckpt is not None and ckpt.exists():                 # resume after an interruption
        c = torch.load(ckpt, map_location=dev, weights_only=False)
        m.load_state_dict(c["model"])
        opt.load_state_dict(c["opt"])
        sched.load_state_dict(c["sched"])
        g.set_state(c["gen"].cpu())
        torch.set_rng_state(c["rng"].cpu())
        torch.cuda.set_rng_state(c["cuda_rng"].cpu())
        trace, start = c["trace"], c["epoch"] + 1
        print(f"  resumed after epoch {c['epoch']}", flush=True)
    t_ep = time.time()
    for ep in range(start, cfg["epochs"] + 1):
        m.train()
        perm = trt[torch.randperm(len(trt), generator=g).to(dev)]
        for i in range(0, per_epoch * bs, bs):
            j = perm[i:i + bs]
            loss = nn.functional.cross_entropy(m(batch(j)), Y[j])
            opt.zero_grad()
            loss.backward()
            opt.step()
            if cfg["per_step"]:
                sched.step()
        if not cfg["per_step"]:
            sched.step()
        if ep in report_at:                              # trajectory only; the result is the last epoch, fixed in advance
            r = evaluate()
            trace[ep] = r[0] if per_class else r
            print(f"  epoch {ep}: {trace[ep]:.4f}", flush=True)
        if ckpt is not None:
            torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "gen": g.get_state(), "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(),
                        "trace": trace, "epoch": ep}, ckpt)
            print(f"  epoch {ep} done, {time.time() - t_ep:.0f} s", flush=True)
            t_ep = time.time()
    out = evaluate()
    return (out, trace) if report_at else out


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


def released(seed=0, split="sessions_out"):
    """One seed at the released training settings on one split (run-disjoint, or random windows with the same
    training size); the two splits run as separate processes and write separate files."""
    PointMLP = load_pointmlp()
    d = load()
    P = frames()
    y5, sp = splits(d, seed)
    t0 = time.time()
    a, b = sp[split]
    ckpt = ROOT / "results" / f"pointmlp_released_s{seed}_{split}.ckpt"     # local only, for resuming
    (acc, pc), trace = run(PointMLP, P, y5, a, b, 5, seed, per_class=True, cfg=RELEASED, report_at=(3, 10, 25, 50, 75),
                           ckpt=ckpt)
    res = {"settings": RELEASED, "seed": seed, "split": split, "acc": acc, "per_subject": pc, "trace": trace,
           "seconds": round(time.time() - t0)}
    print(split, acc, f"{time.time() - t0:.0f}s", flush=True)
    (ROOT / "results" / f"pointmlp_released_s{seed}_{split}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    if sys.argv[1:2] == ["--released"]:
        released(int(sys.argv[2]), sys.argv[3])
    else:
        main(tuple(int(s) for s in sys.argv[1:]) or (0,))
