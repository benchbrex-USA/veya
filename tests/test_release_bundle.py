"""Release bundle, selective thresholds, and ordinal deferral. No checkpoint load."""
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from veya.experimental.ordinal import OrdinalHead  # noqa: E402
from veya.experimental.ordinal_artifact import load_ordinal, save_ordinal  # noqa: E402
from veya.experimental.ordinal_data import OrdinalDataUnavailable, load_ordinal_table  # noqa: E402
from veya.release_bundle import load_router, plan_for, recommend_packed_cutoff, resolve_manifest_path  # noqa: E402
from veya.selective import apply_report, apply_threshold, select_threshold  # noqa: E402
from veya.server import _version_payload  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name if cond else "%s %s" % (name, detail))


def test_cutoff_rule_is_stable():
    grid = [2, 4, 8, 12, 16, 20, 32, 50, 77]
    weak = {n: 0.01 for n in grid}
    weak[77] = 0.03
    cutoff, _reason = recommend_packed_cutoff(weak, grid, current=32)
    check("cutoff/weak 77 keeps 32", cutoff == 32)
    only_77 = {n: 0.0 for n in grid}
    only_77[77] = 0.20
    cutoff, _reason = recommend_packed_cutoff(only_77, grid, current=32)
    check("cutoff/only 77 moves to 50", cutoff == 50, str(cutoff))
    from_32 = {n: 0.0 for n in grid}
    for n in (32, 50, 77):
        from_32[n] = 0.08
    cutoff, _reason = recommend_packed_cutoff(from_32, grid, current=32)
    check("cutoff/win from 32 lowers to 20", cutoff == 20, str(cutoff))


def test_router_plan_and_version_fields():
    router = {
        "packed_max_candidates": 32,
        "hierarchical_k": 8,
        "hierarchical_route": "shortlist_k8_packed_rerank",
    }
    check("plan/4 packed", plan_for(router, 4)["path"] == "packed")
    check("plan/4 calibration", plan_for(router, 4)["calibration"] == "packed")
    wide = plan_for(router, 77)
    check("plan/77 shortlist", wide["path"] == "shortlist" and wide["k"] == 8)
    check("plan/77 route calibration", wide["calibration"] == "shortlist_k8_packed_rerank")
    version = _version_payload(type("A", (), {"cfg": {"model_name": "m", "encoder": "e", "max_len": 1, "head_max_len": 2}})())
    check("version/no shortlist hash", version["shortlist_sha256"] is None)
    check("version/has calibration key", "calibration_version" in version)


def test_threshold_is_chosen_without_the_test_labels():
    scores = np.array([0.9, 0.8, 0.4, 0.2])
    correct = np.array([1, 1, 0, 1])
    threshold = select_threshold(scores, correct, 0.0)
    check("threshold/zero error", abs(threshold - 0.8) < 1e-9, str(threshold))
    held = np.array([0.95, 0.85, 0.5])
    held_correct = np.array([0, 1, 1])
    report = apply_report(held, held_correct, threshold)
    check("threshold/held out accepts two", report["accepted"] == 2, str(report))
    check("threshold/held out risk is one half", abs(report["observed_risk"] - 0.5) < 1e-9)
    check("threshold/none accepts nothing", int(apply_threshold(held, None).sum()) == 0)
    # The test labels above were not used to place the threshold: flipping them
    # does not move a threshold that was already returned.
    check("threshold/frozen", select_threshold(scores, correct, 0.0) == threshold)
    check(
        "threshold/does not split equal-score ties",
        select_threshold(np.array([0.9, 0.9]), np.array([1, 0]), 0.0) is None,
    )


def test_manifest_paths_cannot_escape_root():
    directory = tempfile.mkdtemp()
    try:
        resolve_manifest_path(directory, "../outside.json")
        check("manifest/path traversal rejected", False)
    except ValueError:
        check("manifest/path traversal rejected", True)


def test_shipped_router_matches_the_code_constant():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    router = load_router(os.path.join(root, "models", "production", "choice_router.json"))
    check("shipped/19 packed", plan_for(router, 19)["path"] == "packed")
    check("shipped/20 shortlist", plan_for(router, 20)["path"] == "shortlist" and plan_for(router, 20)["k"] == 8)


def test_ordinal_missing_and_not_production():
    try:
        load_ordinal_table("", ["low", "high"])
        check("ordinal/missing raises", False)
    except OrdinalDataUnavailable:
        check("ordinal/missing raises", True)
    directory = tempfile.mkdtemp()
    path = os.path.join(directory, "ordinal.pt")
    head = OrdinalHead(4, n_levels=3)
    side = save_ordinal(path, head, ["low", "mid", "high"])
    check("ordinal/sidecar not production", side["production"] is False)
    blob = load_ordinal(path)
    check("ordinal/loads research", blob["status"] == "research")
    blob["production"] = True
    torch.save(blob, path)
    try:
        load_ordinal(path)
        check("ordinal/refuses production flag", False)
    except ValueError:
        check("ordinal/refuses production flag", True)
    tiny = os.path.join(directory, "tiny.csv")
    with open(tiny, "w", encoding="utf-8") as handle:
        handle.write("text,label\nhello,low\n")
    try:
        load_ordinal_table(tiny, ["low", "high"])
        check("ordinal/refuses tiny csv", False)
    except ValueError:
        check("ordinal/refuses tiny csv", True)


def test_recorded_checks_pass():
    assert not FAIL, "\n".join(FAIL)


if __name__ == "__main__":
    test_cutoff_rule_is_stable()
    test_router_plan_and_version_fields()
    test_threshold_is_chosen_without_the_test_labels()
    test_manifest_paths_cannot_escape_root()
    test_shipped_router_matches_the_code_constant()
    test_ordinal_missing_and_not_production()
    print("%d passed, %d failed" % (len(PASS), len(FAIL)))
    for item in FAIL:
        print(" FAIL", item)
    raise SystemExit(1 if FAIL else 0)
