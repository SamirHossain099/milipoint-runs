"""Claims of the MiliPoint split audit, checked against results/ (standing rule 4: never a literal)."""
import json
import os
from pathlib import Path

import pytest

R = Path(os.environ.get("RESULTS_DIR", Path(__file__).resolve().parents[1] / "results"))


def load(name):
    f = R / name
    if not f.exists():
        pytest.skip(f"{name} not produced yet")
    return json.loads(f.read_text())


def test_official_split_leaks_almost_every_window():
    g = load("gate.json")["official"]
    assert g["frac_test_with_train_twin_sharing_4"] >= 0.95 and g["frac_test_sharing_any_frame"] == 1.0
    assert g["time_neighbour_iden_acc"] == 1.0
    assert g["time_neighbour_kp_mle_cm"] < min(load("gate.json")["published"]["kp9_mle_cm"]) / 2


def test_keypoints_barely_depend_on_the_leak():
    g = load("gate.json")
    base = g["official"]["mlp_kp_mle_cm"]
    for k in ("kp_subjects_out_9_10", "kp_subjects_out_3_5", "kp_subjects_out_0_8"):
        assert g[k]["mlp_kp_mle_cm"] / base < 1.25, k              # below the 25% GO bar


def test_identification_split_in_subject_order_leaks_the_same():
    g = load("gate.json")["official_iden"]
    assert g["frac_test_with_train_twin_sharing_4"] >= 0.95 and g["time_neighbour_iden_acc"] >= 0.9999   # a boundary window


def test_identification_collapses_on_a_held_out_run_pointnet():
    p = load("pointnet_iden.json")
    assert min(p["official"]) >= 0.80
    assert max(p["sessions_out"]) <= 0.35 and min(p["random_same5"]) - max(p["sessions_out"]) >= 0.25
    assert min(p["random_same5_sizematched"]) - max(p["sessions_out"]) >= 0.25     # not a training-size effect


def test_identification_collapses_with_the_authors_pointmlp():
    for name in ("pointmlp_iden.json", "pointmlp_iden_s1.json"):
        p = load(name)
        assert p["official"][0] >= 0.90 and p["random_same5"][0] - p["sessions_out"][0] >= 0.5, name
        assert p["random_same5_sizematched"][0] - p["sessions_out"][0] >= 0.5, name


def test_shuffled_split_lets_a_no_learning_baseline_reach_the_published_98():
    t = load("action_oracle.json")["trailing"]                  # windows built without labels
    assert t["shuffled"]["input_1nn_acc"] >= 0.98 and t["contiguous"]["input_1nn_acc"] < 0.10


def test_makgcn_split_puts_every_test_window_beside_its_own_performance():
    m = load("action_oracle.json")["makgcn"]
    assert m["shuffled"]["frac_test_same_run_in_train"] == 1.0 and m["contiguous"]["frac_test_same_run_in_train"] == 0.0
    assert m["shuffled"]["input_1nn_acc"] < 0.5                  # the text says the lookup does not reach 98% there


def test_held_out_runs_repeat_the_training_actions():
    h = load("action_oracle.json")["coverage"]["held_out_run_actions"]
    missing = [len(v["not_in_training_runs"]) for v in h.values()]
    assert sorted(missing) == [0, 0, 0, 0, 2]                  # "except two actions of one participant"
    assert min(len(v["actions"]) for v in h.values()) >= 48


def test_label_index_bug_grows_with_the_window():
    b = load("action_bug.json")
    assert 0 < b["stacks_5"]["frac_windows_affected"] < 0.05 < 0.25 < b["stacks_50"]["frac_windows_affected"]


def test_participant_2_held_out_run_is_almost_never_recognised():
    for name in ("pointnet_iden.json", "pointmlp_iden.json", "pointmlp_iden_s1.json"):
        for per_run in load(name)["sessions_out_per_subject"]:
            assert per_run["2"] < 0.01, name                      # label 2 = participant 2 (sorted MULTI keys)


def test_first_pointnet_random_run_is_the_poorly_trained_one():
    p = load("pointnet_iden.json")
    assert p["random_same5"][0] < p["random_same5_sizematched"][0] - 0.2 and min(p["random_same5"][1:]) > 0.8
