"""Phase-2 infrastructure: calibration, server, ordinal head, candidate head, bench protocol."""
import json
import os
import sys
import tempfile
import threading
from urllib.request import Request, urlopen

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from veya.agent import Agent  # noqa: E402
from veya.calibration import (  # noqa: E402
    apply_calibration, build_artifact, dump_calibration, fit_temperature,
    load_calibration, nll, resolve_temperature,
)
from veya.experimental.candidate_set import CandidateSetHead  # noqa: E402
from veya.experimental.ordinal import OrdinalHead  # noqa: E402
from veya.server import serve  # noqa: E402
from veya.shortlist import choose_choice_path  # noqa: E402
from research.veyabench.protocol import (  # noqa: E402
    LeakError, SplitLedger, complement_error, mine_hard_negatives, permutation_report,
    recall_at_k, risk_coverage,
)

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


def _logits(n, k, seed=0):
    """Confident logits whose argmax is wrong 35% of the time, so NLL wants T > 1."""
    rng = np.random.default_rng(seed)
    pred = rng.integers(0, k, size=n)
    logits = rng.normal(size=(n, k)) * 0.05
    logits[np.arange(n), pred] += 4.0
    labels = pred.copy()
    flip = rng.random(n) < 0.35
    labels[flip] = (pred[flip] + 1) % k
    return logits, labels


def test_fit_recovers_temperature_on_a_held_out_split():
    logits, labels = _logits(400, 4, seed=1)
    fit_x, fit_y = logits[:200], labels[:200]
    ev_x, ev_y = logits[200:], labels[200:]
    fitted = fit_temperature(fit_x, fit_y)
    check_true("calibration/overconfident logits soften", fitted > 1.5, "t=%s" % fitted)
    check_true(
        "calibration/held-out nll improves on t=1",
        nll(ev_x, ev_y, fitted) < nll(ev_x, ev_y, 1.0),
    )


def test_small_group_is_not_fitted():
    logits, labels = _logits(10, 3, seed=2)
    artifact = build_artifact({"choice:3-5": (logits, labels)}, "test", "hash", min_observations=32)
    check("calibration/tiny group omitted", artifact["cardinality_temperatures"], {})
    check("calibration/fallback global is usable", resolve_temperature(artifact, "choice", 4) > 0, True)


def test_artifact_roundtrip_and_agent_apply():
    logits, labels = _logits(64, 4, seed=3)
    artifact = build_artifact({"choice": (logits, labels), "choice:3-5": (logits, labels)}, "ckpt", "abc", min_observations=32)
    directory = tempfile.mkdtemp()
    path = os.path.join(directory, "calibration.json")
    dump_calibration(artifact, path)
    loaded = load_calibration(path)
    check("calibration/format", loaded["format"], "veya-calibration-v1")
    agent = Agent.__new__(Agent)
    apply_calibration(agent, loaded)
    check_true("calibration/agent version", agent.calibration_version == "abc")
    check_true("calibration/choice temperature applied", abs(agent.temperature[0] - loaded["primitive_temperatures"]["choice"]) < 1e-9)


def test_reject_test_split_artifact():
    path = os.path.join(tempfile.mkdtemp(), "calibration.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"format": "veya-calibration-v1", "fitting_split": "test"}, handle)
    try:
        load_calibration(path)
        check_true("calibration/rejects test split", False)
    except ValueError:
        check_true("calibration/rejects test split", True)


def test_ledger_blocks_leakage():
    ledger = SplitLedger()
    ledger.add("massive", "intent", "massive-intent", "train", "1", True, "base", "train")
    ledger.add("massive", "intent", "massive-intent", "calibration", "2", True, "base", "calibration")
    ledger.add("clinc", "intent", "clinc", "test", "9", False, "unseen", "test")
    ledger.assert_disjoint()
    try:
        ledger.add("massive", "intent", "massive-intent", "test", "1", False, "base", "test")
        check_true("ledger/duplicate row rejected", False)
    except LeakError:
        check_true("ledger/duplicate row rejected", True)
    try:
        ledger.add("massive", "intent", "massive-intent", "test", "3", True, "base", "test")
        check_true("ledger/test cannot be training exposure", False)
    except LeakError:
        check_true("ledger/test cannot be training exposure", True)


def test_recall_and_hard_negatives():
    check("recall@2", recall_at_k([["a", "b", "c"], ["x", "y"]], ["b", "z"], 2), 0.5)
    labels = ["refund_requested", "refund_approved", "weather"]
    vectors = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    mined = mine_hard_negatives(labels, vectors, k=1)
    check("hard-negative/nearest is the sibling refund", mined["refund_requested"], ["refund_approved"])


def test_consistency_and_permutation_metrics():
    check_true("complement/perfect pair", complement_error(0.2, 0.8) < 1e-9)
    check_true("complement/both high is an error", complement_error(0.72, 0.47) > 0.1)
    report = permutation_report({"a": 0.7, "b": 0.3}, [{"a": 0.7, "b": 0.3}, {"a": 0.2, "b": 0.8}])
    check("permutation/one flip in two", report["top1_flip_rate"], 0.5)
    curve = risk_coverage([1, 1, 0], [0.9, 0.5, 0.1])
    check_true("risk/full coverage error", abs(curve["curve"][-1]["error"] - (1 / 3)) < 1e-9)


def test_choice_path_switches_when_packing_drops_markers():
    from veya.shortlist import production_choice_plan
    check("path/20 packed", choose_choice_path(20), "packed")
    check("path/100 packed", choose_choice_path(100), "packed")
    check("path/150 shortlist", choose_choice_path(150), "shortlist")
    check("plan/19 packed", production_choice_plan(19), {"path": "packed", "k": None})
    check("plan/20 shortlist k8", production_choice_plan(20), {"path": "shortlist", "k": 8})
    check("plan/77 shortlist k8", production_choice_plan(77), {"path": "shortlist", "k": 8})


def test_candidate_head_is_order_equivariant_and_learns():
    torch.manual_seed(0)
    head = CandidateSetHead(8)
    state = torch.randn(2, 8)
    cands = torch.randn(2, 5, 8)
    scores = head(state, cands)
    flipped = head(state, cands.flip(1))
    check_true("candidate/flip swaps scores", torch.allclose(scores.flip(1), flipped, atol=1e-5))
    loss = scores.square().mean()
    loss.backward()
    check_true("candidate/gradients flow", head.score.weight.grad is not None and head.score.weight.grad.abs().sum() > 0)


def test_ordinal_distribution_is_ordered_and_normalized():
    torch.manual_seed(0)
    head = OrdinalHead(4, n_levels=4)
    hidden = torch.randn(6, 4)
    cumulative = head.cumulative_probability(hidden)
    check_true("ordinal/cumulative does not increase", bool(torch.all(cumulative[:, 1:] <= cumulative[:, :-1] + 1e-5)))
    dist = head.distribution(hidden)
    check_true("ordinal/rows sum to 1", torch.allclose(dist.sum(-1), torch.ones(6), atol=1e-4))
    check_true("ordinal/no negative mass", bool(torch.all(dist >= -1e-5)))


class _Fake:
    cfg = {"model_name": "rl-agent", "encoder": "answerdotai/ModernBERT-large", "max_len": 512, "head_max_len": 192}
    calibration_version = "checkpoint"

    def system_one(self, state, questions):
        if not questions:
            return {"model": "veya-rl-agent", "decision": "defer", "reason": "invalid_schema", "answers": {}, "usage": {"input_tokens": 0, "output_tokens": 0}}
        return {
            "model": "veya-rl-agent",
            "decision": "answer",
            "reason": None,
            "answers": {"q": {"type": "noul", "noul": 0.25, "confidence": 0.75, "probabilities": None}},
            "usage": {"input_tokens": 4, "output_tokens": 0},
        }


def test_http_server():
    server = serve(_Fake(), port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        health = json.loads(urlopen("http://127.0.0.1:%d/health" % port).read().decode())
        check("server/health", health["status"], "ok")
        version = json.loads(urlopen("http://127.0.0.1:%d/version" % port).read().decode())
        check("server/version checkpoint", version["checkpoint"], "rl-agent")
        request = Request(
            "http://127.0.0.1:%d/v1/noul" % port,
            data=json.dumps({"state": {"message": "hello"}, "instructions": "Is this a refund?"}).encode(),
            headers={"Content-Type": "application/json"},
        )
        body = json.loads(urlopen(request).read().decode())
        check("server/noul deferred", body["deferred"], False)
        check("server/noul answer", body["answer"], 0.25)
    finally:
        server.shutdown()


def test_recorded_checks_pass():
    assert not FAIL, "\n".join(FAIL)


if __name__ == "__main__":
    test_fit_recovers_temperature_on_a_held_out_split()
    test_small_group_is_not_fitted()
    test_artifact_roundtrip_and_agent_apply()
    test_reject_test_split_artifact()
    test_ledger_blocks_leakage()
    test_recall_and_hard_negatives()
    test_consistency_and_permutation_metrics()
    test_choice_path_switches_when_packing_drops_markers()
    test_candidate_head_is_order_equivariant_and_learns()
    test_ordinal_distribution_is_ordered_and_normalized()
    test_http_server()
    print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
    for item in FAIL:
        print("  FAIL", item)
    raise SystemExit(1 if FAIL else 0)
