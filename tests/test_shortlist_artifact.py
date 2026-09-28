"""Shortlist artifact round-trip and incompatible-checkpoint refusal."""
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from veya.shortlist_artifact import load_shortlist, save_route_calibration, load_route_calibration, save_shortlist  # noqa: E402

PASS, FAIL = [], []


def check_true(name, cond, detail=""):
    (PASS if cond else FAIL).append(name if cond else "%s %s" % (name, detail))


def test_roundtrip_and_refusal():
    torch.manual_seed(0)
    layer = torch.nn.Linear(6, 4, bias=False)
    directory = tempfile.mkdtemp()
    path = os.path.join(directory, "shortlist_k8.pt")
    meta = save_shortlist(path, layer, ["a", "b"], ["intent a", "intent b"], "abc", k=8, seed=1)
    loaded, blob = load_shortlist(path, "abc", k=8)
    probe = torch.randn(1, 6)
    with torch.no_grad():
        same = torch.allclose(layer(probe), loaded(probe), atol=1e-6)
    check_true("artifact/scores match", same)
    check_true("artifact/sidecar hash", len(meta["file_sha256"]) == 64)
    try:
        load_shortlist(path, "different", k=8)
        check_true("artifact/rejects other checkpoint", False)
    except ValueError:
        check_true("artifact/rejects other checkpoint", True)
    cal = os.path.join(directory, "calibration_shortlist_k8.json")
    save_route_calibration(
        cal, 1.25, 40, meta["file_sha256"], "abc", "calibration",
        fit_split_hash="1" * 64,
    )
    loaded_cal = load_route_calibration(cal, "abc", meta["file_sha256"])
    check_true("route-cal/temperature", abs(loaded_cal["temperature"] - 1.25) < 1e-9)
    try:
        load_route_calibration(cal, "abc", "0" * 64)
        check_true("route-cal/rejects other shortlist", False)
    except ValueError:
        check_true("route-cal/rejects other shortlist", True)


def test_production_shortlist_keeps_label_only_semantics():
    """Phase-8 rich descriptions were rejected and must not leak into serving."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    blob = torch.load(
        os.path.join(root, "models", "production", "shortlist_k8.pt"),
        map_location="cpu",
        weights_only=False,
    )
    expected = ["banking intent: " + label.replace("_", " ") for label in blob["labels"]]
    check_true("production/label-only shortlist text", blob["label_texts"] == expected)


def test_recorded_checks_pass():
    assert not FAIL, "\n".join(FAIL)


if __name__ == "__main__":
    test_roundtrip_and_refusal()
    test_production_shortlist_keeps_label_only_semantics()
    print("%d passed, %d failed" % (len(PASS), len(FAIL)))
    for item in FAIL:
        print(" FAIL", item)
    raise SystemExit(1 if FAIL else 0)
