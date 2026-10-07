"""Expert routing, sweep gates, and the audited baseline reader."""

from __future__ import annotations

import json
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from experts.baseline import PHASE10, audited_choice_baseline, snapshot
from experts.evalset import load_dataset, score_row, summarize
from experts.promotions import GATES, classify, matrix
from experts.distill import raw_teacher_output_is_trainable, teacher_record
from experts.evaluate import by_system, disagreement_priority, pareto
from experts.health import Circuit
from experts.registry import EXPENSIVE, ISING, KIMI, LIGHTNING, ULTRA, VEYA, public_registry
from experts.router import route
from experts.sweep import ACT_HEAD, cartesian_size, judge, staged_plan

ROOT = Path(__file__).resolve().parents[1]
ALL = {
    "nvidia/nemotron-3.5-lightning-30b-a3b",
    "deepseek-ai/deepseek-v4.1-flash",
    "z-ai/glm-5.3-flash",
    "openai/gpt-oss-20b",
    "moonshotai/kimi-k3",
    "nvidia/nemotron-3-ultra-550b-a55b",
    "nvidia/ising-calibration-1.5-31b",
    "nvidia/llama-3.1-nemotron-safety-guard-8b-v3",
}


def test_registry_roles_and_unturned_profiles():
    rows = {item["model_id"]: item for item in public_registry()}
    assert rows[VEYA]["role"] == "decision"
    assert rows[ISING]["tasks"] == ["quantum_calibration"]
    assert rows[ISING]["profile"]["temperature"] == 0.2
    assert rows[LIGHTNING]["profile"]["temperature"] == 1.0
    assert rows[LIGHTNING]["profile"]["top_p"] == 0.95
    assert rows["openai/gpt-oss-20b"]["profile"]["top_p"] == 1.0
    assert rows[KIMI]["cost_class"] == "high"
    assert all(item["profile"]["benchbrex_tuned"] is False for item in rows.values())
    assert rows["nvidia/llama-3.1-nemotron-safety-guard-8b-v3"]["safety_only"] is True
    assert rows["nvidia/llama-3.1-nemotron-safety-guard-8b-v3"]["scientific_judge"] is False


def test_router_keeps_veya_and_specialists_in_their_lanes():
    bounded = route({"task": "bounded_decision"}, ALL)
    assert bounded["selected"] == VEYA
    assert bounded["invokes_external"] is False
    quantum = route({"task": "quantum_calibration", "tier": "max", "budget": "frontier"}, ALL)
    assert quantum["selected"] == ISING
    assert "deepseek-ai/deepseek-v4.1-flash" not in quantum["fallback"]
    missing = route({"task": "quantum_calibration"}, set())
    assert missing["selected"] == "defer"
    text = route({"task": "general_text"}, ALL)
    assert text["selected"] == LIGHTNING
    private = route({"task": "general_text", "privacy": "private"}, ALL)
    assert private["selected"] == "openai/gpt-oss-20b"
    assert LIGHTNING not in private["fallback"]
    fast_image = route({"task": "multimodal", "modality": "image", "tier": "fast"}, ALL)
    assert fast_image["selected"] == "deepseek-ai/deepseek-v4.1-flash"
    deep_image = route({"task": "multimodal", "modality": "multimodal", "tier": "deep"}, ALL)
    assert deep_image["selected"] == "z-ai/glm-5.3-flash"
    assert KIMI not in deep_image["fallback"]
    hard_image = route(
        {"task": "multimodal", "modality": "multimodal", "tier": "max", "budget": "frontier"},
        ALL,
    )
    assert hard_image["selected"] == KIMI
    assert hard_image["profile"]["top_p"] == 1.0
    assert hard_image["profile"]["reasoning_effort"] == "max"
    cheap = route(
        {"task": "multimodal", "modality": "image", "tier": "max", "budget": "low"},
        ALL,
    )
    assert cheap["selected"] not in EXPENSIVE
    planning = route({"task": "planning", "tier": "max", "budget": "frontier"}, ALL)
    assert planning["selected"] == ULTRA
    ordinary = route({"task": "planning", "tier": "standard"}, ALL)
    assert ordinary["selected"] == LIGHTNING
    safety = route({"task": "safety"}, ALL)
    assert safety["selected"] == "nvidia/llama-3.1-nemotron-safety-guard-8b-v3"
    assert safety["role"] == "safety"


def test_caller_cannot_name_a_model():
    try:
        route({"task": "general_text", "model": ULTRA}, ALL)
    except ValueError as exc:
        assert "model" in str(exc)
    else:
        raise AssertionError("model override was accepted")


def test_baseline_is_the_audited_file_and_phase10_citation():
    choice = audited_choice_baseline(ROOT)
    assert choice["status"] == "previously_measured_not_rerun"
    assert choice["recall_at_8"] == 0.356
    assert choice["end_to_end_accuracy"] == 0.349
    assert choice["retrieval_misses"] == 1932
    assert choice["recall_at_1"] is None
    report = (ROOT / "docs/phase10/final_report.md").read_text(encoding="utf-8")
    assert "34.97%" in report
    assert "5.33%" in report
    assert "43.10%" in report
    assert PHASE10["suffix_recall_at_8"]["long_irrelevant"] == 0.0533
    record = snapshot(ROOT)
    assert record["optimal"] is False
    assert record["act_head"]["reference_act_weight"] == 0.0
    assert ACT_HEAD["trained_by_this_change"] is False


def test_sweep_is_staged_and_does_not_declare_an_optimum():
    plan = staged_plan()
    staged = sum(len(items) for items in plan.values())
    assert staged < cartesian_size()
    assert all(item["act_weight"] == 0.0 and item["k"] == 8 for items in plan.values() for item in items)
    assert all(item["writes_production_checkpoint"] is False for items in plan.values() for item in items)
    rejected = judge({"split": "blind", "recall_at_8": 0.9, "baseline_recall_at_8": 0.3, "k": 8})
    assert rejected["validation_finalist"] is False
    assert "blind_split_rejected" in rejected["reasons"]
    finalist = judge(
        {
            "split": "validation",
            "k": 8,
            "baseline_recall_at_8": 0.35,
            "recall_at_8": 0.42,
            "long_suffix_recall_at_8": 0.30,
            "seed_recalls": [0.41, 0.40],
            "wrote_production_checkpoint": False,
        }
    )
    assert finalist["validation_finalist"] is True
    assert finalist["optimal"] is False
    assert finalist["promoted_checkpoint"] is False


def test_teacher_rows_and_pareto_and_circuit():
    try:
        teacher_record(
            teacher_model=LIGHTNING,
            teacher_version="baseline",
            prompt_version="v1",
            sampling={"temperature": 1.0},
            teacher_output={"label": "a"},
            accepted_label=None,
            human_verified=False,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("unverified teacher output was accepted")
    record = teacher_record(
        teacher_model=LIGHTNING,
        teacher_version="baseline",
        prompt_version="v1",
        sampling={"temperature": 1.0},
        teacher_output={"label": "a"},
        accepted_label={"label": "b"},
        human_verified=True,
    )
    assert record["teacher_is_ground_truth"] is False
    assert raw_teacher_output_is_trainable(record) is True
    assert record["teacher_sha256"] != record["label_sha256"]
    summaries = by_system(
        [
            {"system": "huge", "correct": True, "latency_ms": 1000, "cost": 5, "external_calls": 1},
            {"system": "small", "correct": True, "latency_ms": 20, "cost": 0.1, "external_calls": 0},
        ]
    )
    assert pareto(summaries) == ["small"]
    assert disagreement_priority(["ising_vs_veya"])["automation"] == "defer"
    circuit = Circuit()
    circuit.record(False)
    circuit.record(False)
    assert circuit.allow() is True
    circuit.record(False)
    assert circuit.allow() is False
    circuit.record(True)
    assert circuit.allow() is True


def test_expert_route_stays_off_the_public_gateway():
    from experts.router import route
    from web import server as gateway

    decision = route({"task": "bounded_decision"})
    assert decision["selected"] == "veya-core"
    assert decision["invokes_external"] is False
    gateway.rate_hits.clear()
    server = ThreadingHTTPServer(("127.0.0.1", 0), gateway.Handler)
    port = server.server_address[1]
    import threading

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        body = json.dumps({"task": "bounded_decision"}).encode()
        request = Request(
            "http://127.0.0.1:%d/api/experts/route" % port,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urlopen(request, timeout=3)
            raise AssertionError("expert route must not be served")
        except HTTPError as exc:
            assert exc.code == 404
    finally:
        server.shutdown()
        server.server_close()


def test_evalset_hash_and_deterministic_scores():
    loaded = load_dataset(ROOT)
    assert loaded["manifest"]["rows"] == 21
    assert loaded["manifest"]["llm_judge"] is False
    families = {row["task_family"] for row in loaded["rows"]}
    assert families == {
        "text_decision",
        "general_reasoning",
        "coding",
        "multimodal",
        "quantum",
        "safety",
    }
    row = loaded["rows"][0]
    assert score_row(row, {"label": "billing"})["correct"] is True
    assert score_row(row, {"label": "technical"})["correct"] is False
    json_row = next(item for item in loaded["rows"] if item["scoring"] == "json_keys")
    assert score_row(json_row, {"object": {"team": "billing", "urgent": False}})["correct"] is True
    flag_row = next(item for item in loaded["rows"] if item["id"] == "safety-injection")
    assert score_row(flag_row, {"flagged": True})["correct"] is True
    assert summarize([{"correct": True}, {"correct": False}])["accuracy"] == 0.5


def test_promotion_matrix_promotes_nothing():
    payload = matrix()
    assert payload["promoted_count"] == 0
    assert "PROMOTED" not in {item["status"] for item in payload["items"]}
    rejected = [item for item in payload["items"] if item["name"] == "noise-augmented-projection"]
    assert rejected and rejected[0]["status"] == "REJECTED"
    assert classify({gate: True for gate in GATES}) == "PROMOTED"
    assert classify({"rejected": True, "full_tests_pass": True}) == "REJECTED"


def test_sources_do_not_contain_a_live_credential():
    for path in (ROOT / "experts").rglob("*.py"):
        assert "nvapi-" not in path.read_text(encoding="utf-8")
