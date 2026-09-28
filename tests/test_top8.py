"""Regression tests for the experimental Top-8 interaction scorer (veya/experimental/top8.py).

Run: python tests/test_top8.py
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from veya.experimental.top8 import Top8Interaction  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name if cond else "%s %s" % (name, detail))


def test_forward_shapes():
    model = Top8Interaction(dim=32, width=64).eval()
    query = torch.randn(3, 32)
    candidates = torch.randn(3, 8, 32)
    with torch.no_grad():
        logits = model(query, candidates)
    check("top8/shape [B,8]", tuple(logits.shape) == (3, 8), str(tuple(logits.shape)))
    check("top8/finite", bool(torch.isfinite(logits).all()))


def test_deterministic_and_discriminating():
    torch.manual_seed(0)
    model = Top8Interaction(dim=32, width=64).eval()
    query = torch.randn(1, 32)
    candidates = torch.randn(1, 8, 32)
    with torch.no_grad():
        first = model(query, candidates)
        again = model(query, candidates)
        changed = model(query, candidates + 0.5)
    check("top8/deterministic", bool(torch.equal(first, again)))
    check("top8/different candidates differ", bool((first != changed).any()))


def test_single_batch_entry():
    model = Top8Interaction(dim=16, width=32).eval()
    with torch.no_grad():
        logits = model(torch.randn(1, 16), torch.randn(1, 8, 16))
    check("top8/single entry shape", tuple(logits.shape) == (1, 8))


def test_state_dict_round_trip():
    torch.manual_seed(1)
    model = Top8Interaction(dim=32, width=64).eval()
    query = torch.randn(1, 32)
    candidates = torch.randn(1, 8, 32)
    with torch.no_grad():
        before = model(query, candidates)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "top8.pt"
        torch.save(model.state_dict(), path)
        twin = Top8Interaction(dim=32, width=64).eval()
        twin.load_state_dict(torch.load(path))
        with torch.no_grad():
            after = twin(query, candidates)
    check("top8/round trip same outputs", bool(torch.equal(before, after)))


def test_recorded_checks_pass():
    assert not FAIL, "\n".join(FAIL)


if __name__ == "__main__":
    test_forward_shapes()
    test_deterministic_and_discriminating()
    test_single_batch_entry()
    test_state_dict_round_trip()
    print("%d passed, %d failed" % (len(PASS), len(FAIL)))
    for item in FAIL:
        print(" FAIL", item)
    raise SystemExit(1 if FAIL else 0)
