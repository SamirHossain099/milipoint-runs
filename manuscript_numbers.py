"""Every number the paper quotes, from results/ into results/manuscript_numbers.json (standing rule 4).

    python manuscript_numbers.py && python render_manuscript.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
from pointnet_iden import MULTI  # noqa: E402

ROOT = Path(__file__).resolve().parent
R = ROOT / "results"


def j(n):
    return json.loads((R / n).read_text())


def pct(v, d=1):
    return f"{100 * v:.{d}f}"


def pointmlp():
    """The two PointMLP runs (seed 0 and seed 1) as one result with a list per split."""
    a, b = j("pointmlp_iden.json"), j("pointmlp_iden_s1.json")
    return {k: a[k] + b[k] for k in a if isinstance(a[k], list)} | {"published_pointmlp_iden": a["published_pointmlp_iden"]}


def per_run(res):
    """Accuracy per held-out run (participant), mean over seeds."""
    subj = res["sessions_out_per_subject"]
    return {c: float(np.mean([s[c] for s in subj])) for c in subj[0]}


def main():
    g, pn, pm = j("gate.json"), j("pointnet_iden.json"), pointmlp()
    ao, bug = j("action_oracle.json"), j("action_bug.json")
    N = {}

    def put(k, v, text):
        N[k] = {"value": v, "text": text}

    def span(k, vals, d=1):
        put(f"{k}_min", min(vals), pct(min(vals), d))
        put(f"{k}_max", max(vals), pct(max(vals), d))

    o = g["official"]
    put("n_windows", g["n_windows"], f"{g['n_windows']:,}")
    put("n_test", o["n_test"], f"{o['n_test']:,}")
    put("twin4", o["frac_test_with_train_twin_sharing_4"], pct(o["frac_test_with_train_twin_sharing_4"], 0))
    oi = g["official_iden"]
    put("twin4_iden", oi["frac_test_with_train_twin_sharing_4"], pct(oi["frac_test_with_train_twin_sharing_4"], 0))
    put("tn_kp", o["time_neighbour_kp_mle_cm"], f"{o['time_neighbour_kp_mle_cm']:.1f}")
    put("tn_id", oi["time_neighbour_iden_acc"], pct(oi["time_neighbour_iden_acc"], 0))
    put("pub_kp_best", min(g["published"]["kp9_mle_cm"]), f"{min(g['published']['kp9_mle_cm']):.2f}")
    put("pub_id_best", max(g["published"]["iden_acc"]), f"{max(g['published']['iden_acc']):.2f}")
    put("pub_pm", pm["published_pointmlp_iden"] / 100, f"{pm['published_pointmlp_iden']:.1f}")
    kp = [g[k]["mlp_kp_mle_cm"] for k in ("kp_subjects_out_9_10", "kp_subjects_out_3_5", "kp_subjects_out_0_8")]
    put("kp_mlp_off", o["mlp_kp_mle_cm"], f"{o['mlp_kp_mle_cm']:.1f}")
    put("kp_mlp_out_min", min(kp), f"{min(kp):.1f}")
    put("kp_mlp_out_max", max(kp), f"{max(kp):.1f}")
    put("kp_rise_max", max(kp) / o["mlp_kp_mle_cm"] - 1, pct(max(kp) / o["mlp_kp_mle_cm"] - 1, 0))

    # identification: released split (subject order), random windows of five, size-matched, held-out run
    for m, r in (("pm", pm), ("pn", pn)):
        span(f"{m}_off", r["official"])
        span(f"{m}_r5", r["random_same5"])
        span(f"{m}_sm", r["random_same5_sizematched"])
        span(f"{m}_run", r["sessions_out"])
        pr = per_run(r)
        put(f"{m}_subj_min", min(pr.values()), pct(min(pr.values())))
        put(f"{m}_subj_max", max(pr.values()), pct(max(pr.values()), 0))
        put(f"{m}_per_run", pr, ", ".join(pct(v, 0) for v in pr.values()))
    p2 = sorted(MULTI).index(2)                           # label of participant 2 among the five
    p2_all = [d[str(p2)] for r in (pm, pn) for d in r["sessions_out_per_subject"]]
    put("p2_max", max(p2_all), pct(max(p2_all)))
    put("pn_sm_s0", pn["random_same5_sizematched"][0], pct(pn["random_same5_sizematched"][0]))
    put("pm_r5_int", float(np.mean(pm["random_same5"])), pct(np.mean(pm["random_same5"]), 0))
    put("pm_run_int", float(np.mean(pm["sessions_out"])), pct(np.mean(pm["sessions_out"]), 0))

    # actions
    t, mk = ao["trailing"], ao["makgcn"]
    put("act_1nn_shuf", t["shuffled"]["input_1nn_acc"], pct(t["shuffled"]["input_1nn_acc"]))
    put("act_tn_shuf", t["shuffled"]["time_neighbour_acc"], pct(t["shuffled"]["time_neighbour_acc"]))
    put("act_1nn_off", t["contiguous"]["input_1nn_acc"], pct(t["contiguous"]["input_1nn_acc"]))
    put("act_chance", ao["chance"], pct(ao["chance"]))
    put("act_1nn_shuf_int", t["shuffled"]["input_1nn_acc"], pct(t["shuffled"]["input_1nn_acc"], 0))
    put("mak_n_test", mk["shuffled"]["n_test"], f"{mk['shuffled']['n_test']:,}")
    put("mak_tn", mk["shuffled"]["time_neighbour_acc"], pct(mk["shuffled"]["time_neighbour_acc"]))
    put("mak_1nn", mk["shuffled"]["input_1nn_acc"], pct(mk["shuffled"]["input_1nn_acc"]))
    put("mak_same_run", mk["shuffled"]["frac_test_same_run_in_train"], pct(mk["shuffled"]["frac_test_same_run_in_train"], 0))
    for s in ("5", "50"):
        b = bug[f"stacks_{s}"]
        put(f"bug{s}_windows", b["frac_windows_affected"], pct(b["frac_windows_affected"]))
        put(f"bug{s}_foreign", b["frac_windows_with_foreign_action"], pct(b["frac_windows_with_foreign_action"]))

    # label coverage
    c = ao["coverage"]
    put("lab_frac", c["frac_frames_labelled"], pct(c["frac_frames_labelled"], 0))
    nz = [v for v in c["n_actions_per_file"].values() if v]
    put("cov_min", min(nz), str(min(nz)))
    put("cov_max", max(nz), str(max(nz)))
    h = list(c["held_out_run_actions"].values())
    put("held_min", min(len(x["actions"]) for x in h), str(min(len(x["actions"]) for x in h)))
    put("held_max", max(len(x["actions"]) for x in h), str(max(len(x["actions"]) for x in h)))
    miss = [len(x["not_in_training_runs"]) for x in h]
    assert sum(1 for m in miss if m) == 1, "the text says one participant"
    put("held_missing", max(miss), ["none", "one", "two", "three"][max(miss)])

    # Table 1: identification accuracy (%) by model and split
    def vals(r, k):
        return ", ".join(pct(v) for v in r[k])
    rows = ["| Split | Test data | PointMLP (authors' model) | PointNet | Chance |", "|---|---|---|---|---|",
            f"| Released (shuffled windows) | random windows, 11 participants | {vals(pm, 'official')} | "
            f"{vals(pn, 'official')} | 9.1 |",
            f"| Random windows, 5 participants | random windows | {vals(pm, 'random_same5')} | "
            f"{vals(pn, 'random_same5')} | 20.0 |",
            f"| Random windows, 5 participants, smaller training set | random windows | "
            f"{vals(pm, 'random_same5_sizematched')} | {vals(pn, 'random_same5_sizematched')} | 20.0 |",
            f"| Held-out run, 5 participants | one whole run per participant | {vals(pm, 'sessions_out')} | "
            f"{vals(pn, 'sessions_out')} | 20.0 |"]
    put("table1", None, "\n".join(rows))
    # Table 2: keypoints (MLE, cm) and actions (accuracy, %) under the released and the alternative splits
    folds = (("kp_subjects_out_9_10", "participants 9, 10 out"), ("kp_subjects_out_3_5", "participants 3, 5 out"),
             ("kp_subjects_out_0_8", "participants 0, 8 out"))
    rows = ["| Task | Split | Nearest training window in time | 1-NN on window features | Perceptron on window features |",
            "|---|---|---|---|---|",
            f"| Keypoints (MLE, cm) | released (shuffled windows) | {o['time_neighbour_kp_mle_cm']:.1f} | "
            f"{o['nn1_kp_mle_cm']:.1f} | {o['mlp_kp_mle_cm']:.1f} |"]
    for k, lab in folds:
        rows.append(f"| Keypoints (MLE, cm) | {lab} | {g[k]['time_neighbour_kp_mle_cm']:.1f} | {g[k]['nn1_kp_mle_cm']:.1f} | "
                    f"{g[k]['mlp_kp_mle_cm']:.1f} |")
    tc = t["contiguous"]["time_neighbour_acc"]
    rows += [f"| Actions (accuracy, %) | released (contiguous), 50 frames | "
             f"{'none within 5,000 windows' if tc is None else pct(tc)} | {pct(t['contiguous']['input_1nn_acc'])} | n/a |",
             f"| Actions (accuracy, %) | 50 frames, shuffled, 8:1:1 | {pct(t['shuffled']['time_neighbour_acc'])} | "
             f"{pct(t['shuffled']['input_1nn_acc'])} | n/a |",
             f"| Actions (accuracy, %) | MAK-GCN code (one-action windows, shuffled) | "
             f"{pct(mk['shuffled']['time_neighbour_acc'])} | {pct(mk['shuffled']['input_1nn_acc'])} | n/a |"]
    put("table2", None, "\n".join(rows))
    (R / "manuscript_numbers.json").write_text(json.dumps(N, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(N)} numbers")


if __name__ == "__main__":
    main()
