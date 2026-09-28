"""Correctness regressions that do not need a downloaded checkpoint.

Covers the zero-budget truncate, ECE at confidence 0, JSON-safe routing, and the
language gate that stops a Latin-only checkpoint from answering a script it cannot read.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from veya.agent import Agent  # noqa: E402
from veya.common import build_sequence, ece_score, safe_automation_coverage  # noqa: E402
from veya.lang import defer_reason  # noqa: E402
from veya.router import Router  # noqa: E402

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s:\n     got  %r\n     want %r" % (name, got, want))


def check_true(name, cond, detail=""):
    if cond:
        PASS.append(name)
    else:
        FAIL.append("%s %s" % (name, detail))


class Tok:
    mask_token = "[MASK]"
    mask_token_id = 4
    cls_token_id = 2
    sep_token_id = 3

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def test_truncate_zero_budget():
    q = {
        "t": "choice",
        "ins": "x" * 40,
        "crit": {"a": "alpha description", "b": "beta description", "c": "gamma description"},
    }
    state = "TAILMARKER"
    ids, _markers = build_sequence(Tok(), state, q, max_len=24, head_max_len=40, truncate_left=True)
    tail = [ord(c) for c in "TAIL"]
    present = any(ids[i:i + 4] == tail for i in range(len(ids)))
    check("truncate/zero budget drops the state", present, False)
    check("truncate/result stays inside the window", len(ids) <= 24, True)


def test_truncate_keeps_the_end_when_there_is_room():
    q = {"t": "noul", "ins": "short?", "crit": None}
    state = "HEAD" + ("x" * 80) + "TAIL"
    ids, _markers = build_sequence(Tok(), state, q, max_len=80, head_max_len=40, truncate_left=True)
    tail = [ord(c) for c in "TAIL"]
    present = any(ids[i:i + 4] == tail for i in range(len(ids)))
    check_true("truncate/keeps the end of a long state", present)


def test_ece_counts_zero_confidence():
    conf = np.array([0.0, 0.0])
    correct = np.array([0.0, 1.0])
    score = ece_score(conf, correct)
    check_true("ece/zero confidence is not reported as perfect", score > 0.4, "ece=%s" % score)


def test_safe_automation_coverage():
    correct = np.array([1, 1, 0, 0])
    confidence = np.array([0.9, 0.8, 0.2, 0.1])
    check("sac/epsilon 0 keeps only the correct prefix", safe_automation_coverage(correct, confidence, 0.0), 0.5)
    check("sac/epsilon 0.5 can keep everything", safe_automation_coverage(correct, confidence, 0.5), 1.0)
    check(
        "sac/does not split equal-confidence ties",
        safe_automation_coverage([1, 0], [0.9, 0.9], 0.0),
        0.0,
    )


def test_auto_route_repo_is_json():
    questions = {
        "action": {"type": "choice", "instructions": "x", "criteria": ["a", "b"]},
        "category": {"type": "choice", "instructions": "x", "criteria": ["a"]},
        "churn_risk": {"type": "noul", "instructions": "x"},
        "needs_human": {"type": "noul", "instructions": "x"},
        "urgency": {"type": "score", "instructions": "x", "criteria": ["low", "high"]},
    }
    decision = Router(auto_task_detection=True).route({"body": "charged twice"}, questions)
    check("route/auto repo type", type(decision["repo"]).__name__, "str")
    check_true("route/auto repo serialises", isinstance(json.dumps(decision), str))


def test_defer_without_loading_weights():
    english = {"encoder": "answerdotai/ModernBERT-large"}
    multi = {"encoder": "jhu-clsp/mmBERT-base"}
    check("defer/hindi on english", defer_reason(english, {"message": "मुझसे दो बार शुल्क लिया गया"}), "unsupported_language")
    check("defer/english text", defer_reason(english, {"message": "Please refund the duplicate charge."}), None)
    check("defer/french on english", defer_reason(english, {"message": "Je veux un remboursement pour ce paiement."}), "unsupported_language")
    check("defer/hindi on multilingual", defer_reason(multi, {"message": "मुझसे दो बार शुल्क लिया गया"}), None)

    bare = Agent.__new__(Agent)
    bare.cfg = english
    empty = bare.system_one({"message": "hello"}, {})
    check("schema/empty questions defer", empty["reason"], "invalid_schema")
    check("schema/empty questions have no answers", empty["answers"], {})
    empty_state = bare.system_one("   ", {"q": {"type": "noul", "instructions": "refund?"}})
    check("schema/empty state defers", empty_state["reason"], "invalid_schema")
    blocked = bare.system_one({"message": "发票被重复扣款"}, {"q": {"type": "noul", "instructions": "refund?"}})
    check("defer/runtime reason", blocked["reason"], "unsupported_language")
    check("defer/runtime does not invent an answer", blocked["answers"], {})


def test_recorded_checks_pass():
    assert not FAIL, "\n".join(FAIL)


if __name__ == "__main__":
    test_truncate_zero_budget()
    test_truncate_keeps_the_end_when_there_is_room()
    test_ece_counts_zero_confidence()
    test_safe_automation_coverage()
    test_auto_route_repo_is_json()
    test_defer_without_loading_weights()
    print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
    for item in FAIL:
        print("  FAIL", item)
    raise SystemExit(1 if FAIL else 0)
