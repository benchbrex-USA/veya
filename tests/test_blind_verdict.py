"""Synthetic verification for research/veyabench/blind_verdict.py.

50 clean + 50 shifted synthetic cases with known correctness pin the verdict
structure, the P1-P6 gate wiring, the Section-16 result-language templates,
and the failure precedence between them. All inputs are synthetic files under
``tmp_path``: no checkpoint, tokenizer, or git state outside the sandbox is
touched.

Run: python -m pytest tests/test_blind_verdict.py -xvs
"""
import json
import math
import os
import pathlib
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from research.veyabench.blind_verdict import (  # noqa: E402
    BLIND_VERDICT_FORMAT,
    KNOWN_SLICE_CONDITIONS,
    perturb_text,
    run_blind_verdict,
    verdict_summary,
)
from research.veyabench.freeze_record import build_freeze_record  # noqa: E402

CLASSES = ("alpha", "beta", "delta", "epsilon", "gamma")  # sorted order
VOCAB = {label: index for index, label in enumerate(CLASSES)}
SHIFT_TAG = "short_suffix"
N_CLEAN = 50
N_SHIFTED = 50

TRAINING_TEXTS = [
    "warehouse inventory reconciliation for the quarterly audit",
    "payroll disbursement schedule for the northern district office",
    "freight manifest consolidation for the harbor authority",
    "vaccine cold-chain inspection for the regional clinic network",
    "orchard irrigation timing for the cooperative farm board",
]

REQUIRED_VERDICT_KEYS = {
    "format",
    "freeze_record_hash",
    "split_provenance",
    "contamination",
    "gates",
    "calibration",
    "primary_gates",
    "all_primary_gates_pass",
    "secondary",
    "result_language",
    "sample_size_check",
}

PRIMARY_GATES = {
    "P1_data_validity",
    "P2_selective_safety",
    "P3_useful_coverage",
    "P4_shift_robustness",
    "P5_calibration_transfer",
    "P6_operational_viability",
}

RISK_LANGUAGE = (
    "The frozen selective policy did not maintain the preregistered risk "
    "bound on prospective traffic."
)
COVERAGE_LANGUAGE = (
    "Veya met the risk target but did not automate enough traffic to meet "
    "the preregistered usefulness threshold."
)
SHIFT_LANGUAGE = (
    "Clean-domain selective performance transferred, but the system did not "
    "satisfy the preregistered robustness requirements under %s." % SHIFT_TAG
)
CONTAMINATION_LANGUAGE = (
    "The blind evaluation data failed the preregistered contamination "
    "screens, so no selective-automation claim is supportable on this traffic."
)


# Four varying word slots chosen so any two distinct indices differ in at
# least two slots: (a, b, c) are the index's digits and d = (a + b + c) mod 10.
# Short texts with no shared boilerplate keep every pairwise similarity far
# below the contamination thresholds, so only an injected exact duplicate
# (test_contamination_failure_beats_every_other_gate) ever flags.
_TEXT_VERBS = (
    "check", "renew", "cancel", "merge", "split",
    "export", "flag", "audit", "review", "archive",
)
_TEXT_ADJECTIVES = (
    "urgent", "routine", "disputed", "annual", "portable",
    "electronic", "manual", "interim", "regional", "premium",
)
_TEXT_AUXILIARIES = ("standard", "priority")
_TEXT_NOUNS = (
    "parcel", "rota", "folio", "invoice", "permit",
    "manifest", "coupon", "receipt", "claim", "voucher",
)


def _text(index):
    a = index % 10
    b = (index // 10) % 10
    c = (index // 100) % 10
    d = (a + b + c) % 10
    return "%s %s %s %s %d" % (
        _TEXT_VERBS[a],
        _TEXT_ADJECTIVES[b],
        _TEXT_AUXILIARIES[c],
        _TEXT_NOUNS[d],
        index,
    )


def _make_case(index, tag, gold, correct, confidence):
    pred = gold if correct else CLASSES[(VOCAB[gold] + 1) % len(CLASSES)]
    probabilities = [0.0] * len(CLASSES)
    probabilities[VOCAB[pred]] = float(confidence)
    others = [i for i in range(len(CLASSES)) if i != VOCAB[pred]]
    share = (1.0 - confidence) / len(others)
    for i in others:
        probabilities[i] = share
    logits = [math.log(p) for p in probabilities]  # softmax(logits) == probabilities
    text = _text(index)
    if tag != "clean":
        text = perturb_text(text, tag, "case-%d" % index)
    return {
        "text": text,
        "gold_label": gold,
        "prediction": pred,
        "correct": bool(correct),
        "probabilities": probabilities,
        "logits": logits,
        "slice_tag": tag,
        "confidence": float(confidence),
        "candidate_count": len(CLASSES),
    }


def _clean_slice(accepted_correct, deferred_correct, deferred_wrong, accepted_wrong=0):
    """Clean cases: high-confidence accepted rows plus low-confidence deferred rows."""
    cases = []
    index = 0
    for _ in range(accepted_correct):
        cases.append(_make_case(index, "clean", CLASSES[index % len(CLASSES)], True, 0.92))
        index += 1
    for _ in range(accepted_wrong):
        cases.append(_make_case(index, "clean", CLASSES[index % len(CLASSES)], False, 0.92))
        index += 1
    for _ in range(deferred_correct):
        cases.append(_make_case(index, "clean", CLASSES[index % len(CLASSES)], True, 0.60))
        index += 1
    for _ in range(deferred_wrong):
        cases.append(_make_case(index, "clean", CLASSES[index % len(CLASSES)], False, 0.60))
        index += 1
    return cases


def _shifted_slice(accepted_correct, deferred, accepted_wrong=0):
    cases = []
    index = N_CLEAN
    for _ in range(accepted_correct):
        cases.append(
            _make_case(index, SHIFT_TAG, CLASSES[index % len(CLASSES)], True, 0.90)
        )
        index += 1
    for _ in range(accepted_wrong):
        cases.append(
            _make_case(index, SHIFT_TAG, CLASSES[index % len(CLASSES)], False, 0.90)
        )
        index += 1
    for _ in range(deferred):
        cases.append(
            _make_case(index, SHIFT_TAG, CLASSES[index % len(CLASSES)], True, 0.55)
        )
        index += 1
    return cases


def _passing_cases():
    # 48 accepted correct, 1 deferred correct, 1 deferred wrong -> coverage 96%.
    return _clean_slice(48, 1, 1) + _shifted_slice(48, 2)


def _freeze_record(root, epsilon, c_min):
    root = pathlib.Path(root)
    root.mkdir(parents=True, exist_ok=True)
    model = root / "model.safetensors"
    model.write_bytes(b"synthetic blind-verdict weights")
    tokenizer = root / "tokenizer"
    tokenizer.mkdir(exist_ok=True)
    (tokenizer / "tokenizer.json").write_text('{"version": "A"}', encoding="utf-8")
    shortlist = root / "shortlist_k8.pt.json"
    shortlist.write_text(json.dumps({"k": 8, "seed": 13}), encoding="utf-8")
    temperature = root / "calibration.json"
    temperature.write_text(json.dumps({"method": "temperature", "T": 1.0}), encoding="utf-8")
    policy_artifact = root / "selective_policy.json"
    policy_artifact.write_text(
        json.dumps({"score": "max_probability", "thresholds": {"0.10": 0.85}}),
        encoding="utf-8",
    )
    return build_freeze_record(
        model_path=model,
        tokenizer_dir=tokenizer,
        git_commit="e852981ae34990f4358979625854259302feaa78",
        scorer_id="phase12-sparse-scorer-v1",
        shortlist_config_path=shortlist,
        calibration_method="temperature scaling on the calibration split",
        temperature_artifact_path=temperature,
        selective_policy_algorithm="max_probability threshold at epsilon",
        selective_policy_artifact_path=policy_artifact,
        epsilon=epsilon,
        c_min=c_min,
        perturbation_definitions_version="phase11-noise-v1",
        metric_implementation_version="veyabench-metrics-v1",
        exclusion_rules=["exclude exact development-set duplicates"],
        random_seeds={"bootstrap": 13},
        bootstrap_draws=1000,
        bootstrap_confidence=0.95,
        production_preprocessing_version="veya-preprocess-v1",
    )


def _policy(epsilon):
    key = "%.2f" % epsilon
    return {
        "format": "veya-selective-policy-v1",
        "route": "choice",
        "checkpoint_hash": "a" * 64,
        "shortlist_sha256": "b" * 64,
        "scorer_id": "phase12-sparse-scorer-v1",
        "chosen_on": "validation",
        "validation_hash": "c" * 64,
        "score": "max_probability",
        "operating_epsilon": key,
        "thresholds": {key: 0.85},
    }


def _config(tmp_path, cases, epsilon=0.10, c_min=0.80, training_texts=None, operational="default"):
    if operational == "default":
        operational = {"latency_p95_ms": 41.5, "throughput_rps": 120, "passed": True}
    return {
        "freeze_record": _freeze_record(tmp_path / "freeze", epsilon, c_min),
        "cases": cases,
        "training_texts": TRAINING_TEXTS if training_texts is None else training_texts,
        "blind_vectors": None,
        "training_vectors": None,
        "policy": _policy(epsilon),
        "epsilon": epsilon,
        "c_min": c_min,
        "coverage_retention_min": 0.95,
        "collapse_floor": {"coverage": 0.05, "accuracy": 0.30},
        "operational": operational,
        "temperature": 1.0,
        "domain": "synthetic-assistance",
    }


def test_verdict_structure_on_the_passing_scenario(tmp_path):
    verdict = run_blind_verdict(_config(tmp_path, _passing_cases()))
    assert verdict["format"] == BLIND_VERDICT_FORMAT == "veya-blind-verdict-v1"
    assert REQUIRED_VERDICT_KEYS <= set(verdict)
    assert len(verdict["freeze_record_hash"]) == 64
    int(verdict["freeze_record_hash"], 16)

    assert set(verdict["primary_gates"]) == PRIMARY_GATES
    assert all(isinstance(value, bool) for value in verdict["primary_gates"].values())
    assert verdict["primary_gates"] == {
        "P1_data_validity": True,
        "P2_selective_safety": True,
        "P3_useful_coverage": True,
        "P4_shift_robustness": True,
        "P5_calibration_transfer": True,
        "P6_operational_viability": True,
    }
    assert verdict["all_primary_gates_pass"] is True
    assert verdict["all_primary_gates_pass"] == all(verdict["primary_gates"].values())
    assert verdict["gates"]["passed"] is True
    assert verdict["gates"]["failed_gates"] == []
    assert verdict["contamination"]["total_flagged"] == []
    assert verdict["calibration"]["summary"]["n_slices"] == 2
    assert verdict["calibration"]["summary"]["confidence_separates_accepted_from_deferred"] is True
    assert verdict["operational"] == {
        "present": True,
        "failures": [],
        "measurements": {"latency_p95_ms": 41.5, "throughput_rps": 120, "passed": True},
    }
    assert verdict["slice_families"] == {
        "clean": "clean",
        SHIFT_TAG: "phase11-generic-noise-v1",
    }
    # The verdict is pure data: it round-trips through JSON unchanged.
    assert json.loads(json.dumps(verdict)) == verdict


def test_result_language_success_template(tmp_path):
    verdict = run_blind_verdict(_config(tmp_path, _passing_cases()))
    language = verdict["result_language"]
    assert language.startswith("On prospectively collected synthetic-assistance traffic")
    assert "preregistered frozen configuration" in language
    assert "automatically handled 96.0%" in language
    assert "observed error rate of 0.0%" in language
    assert "upper confidence bound of 7.4%" in language
    assert language.endswith("Remaining cases were deferred.")
    summary = verdict_summary(verdict)
    assert summary.startswith("PASS")
    assert "96.0%" in summary and "7.4%" in summary


def test_secondary_metrics_are_finite_and_consistent(tmp_path):
    verdict = run_blind_verdict(_config(tmp_path, _passing_cases()))
    secondary = verdict["secondary"]
    for key in ("accuracy", "macro_f1", "ece", "brier"):
        assert isinstance(secondary[key], float), key
        assert math.isfinite(secondary[key]), key
    # 48 accepted correct + 1 deferred correct + 1 deferred wrong.
    assert secondary["accuracy"] == pytest.approx(49 / 50)
    assert 0.9 < secondary["macro_f1"] <= 1.0
    assert secondary["ece"] == pytest.approx(0.0808, abs=1e-9)
    assert secondary["brier"] == pytest.approx(0.03568, abs=1e-6)

    per_class = secondary["per_class"]
    assert set(per_class) == set(CLASSES)
    for label, row in per_class.items():
        assert {"support", "correct", "accuracy", "precision", "recall", "f1"} <= set(row)
        assert row["support"] > 0
        assert 0.0 <= row["accuracy"] <= 1.0
        assert 0.0 <= row["f1"] <= 1.0
    # The single deferred mistake (case 49) is a gold-gamma case predicted as alpha.
    assert per_class["gamma"]["support"] == 10
    assert per_class["gamma"]["accuracy"] == pytest.approx(9 / 10)
    assert per_class["gamma"]["f1"] == pytest.approx(18 / 19)
    assert per_class["alpha"]["f1"] == pytest.approx(20 / 21)
    assert per_class["epsilon"]["support"] == 10
    assert per_class["delta"]["f1"] == 1.0

    bootstrap = secondary["bootstrap_ci"]
    assert bootstrap["baseline"] == "all-correct"
    assert bootstrap["delta"] == pytest.approx(-0.02)
    assert bootstrap["ci95_low"] <= bootstrap["ci95_high"]
    assert bootstrap["accuracy"] == pytest.approx(0.98)
    assert bootstrap["accuracy_ci95_low"] <= secondary["accuracy"] <= bootstrap["accuracy_ci95_high"]
    assert bootstrap["n"] == N_CLEAN and bootstrap["draws"] == 1000


def test_sample_size_check_is_present_and_sufficient(tmp_path):
    verdict = run_blind_verdict(_config(tmp_path, _passing_cases()))
    check = verdict["sample_size_check"]
    assert {"required_accepted", "actual_accepted", "sufficient"} <= set(check)
    assert check["actual_accepted"] == 48
    # Wilson UCB(0, n) = z^2 / (n + z^2) <= 0.10 first holds at n = 35.
    assert check["required_accepted"] == 35
    assert check["sufficient"] is True


def test_run_once_is_deterministic(tmp_path):
    config = _config(tmp_path, _passing_cases())
    assert run_blind_verdict(config) == run_blind_verdict(config)


def test_high_risk_cases_fail_p2_with_risk_language(tmp_path):
    cases = _clean_slice(38, 1, 1, accepted_wrong=10) + _shifted_slice(48, 2)
    verdict = run_blind_verdict(_config(tmp_path, cases))
    assert verdict["primary_gates"]["P2_selective_safety"] is False
    assert verdict["primary_gates"]["P1_data_validity"] is True
    assert verdict["primary_gates"]["P3_useful_coverage"] is True
    assert verdict["primary_gates"]["P4_shift_robustness"] is True
    assert verdict["all_primary_gates_pass"] is False
    assert verdict["all_primary_gates_pass"] == all(verdict["primary_gates"].values())
    assert verdict["gates"]["gates"]["G1_clean_safety"]["passed"] is False
    assert verdict["gates"]["slices"]["clean"]["incorrect"] == 10
    assert verdict["result_language"] == RISK_LANGUAGE
    summary = verdict_summary(verdict)
    assert summary.startswith("FAIL")
    assert "P2_selective_safety" in summary


def test_shift_failures_produce_shift_language(tmp_path):
    cases = _clean_slice(48, 1, 1) + _shifted_slice(38, 2, accepted_wrong=10)
    verdict = run_blind_verdict(_config(tmp_path, cases))
    assert verdict["primary_gates"]["P2_selective_safety"] is True
    assert verdict["primary_gates"]["P3_useful_coverage"] is True
    assert verdict["primary_gates"]["P4_shift_robustness"] is False
    assert verdict["gates"]["gates"]["G2_shift_safety"]["failing_slices"] == [SHIFT_TAG]
    assert verdict["result_language"] == SHIFT_LANGUAGE


def test_low_coverage_produces_usefulness_language(tmp_path):
    # 30 accepted of 50 -> coverage 60% < c_min 80%; UCB(0, 30) ~ 0.114 <= 0.15,
    # so the risk gates pass and only usefulness fails.
    clean = [
        _make_case(i, "clean", CLASSES[i % len(CLASSES)], True, 0.97)
        for i in range(30)
    ] + [
        _make_case(30 + i, "clean", CLASSES[i % len(CLASSES)], True, 0.60)
        for i in range(20)
    ]
    shifted = [
        _make_case(100 + i, SHIFT_TAG, CLASSES[i % len(CLASSES)], True, 0.97)
        for i in range(30)
    ] + [
        _make_case(130 + i, SHIFT_TAG, CLASSES[i % len(CLASSES)], True, 0.55)
        for i in range(20)
    ]
    config = _config(tmp_path, clean + shifted, epsilon=0.15, c_min=0.80)
    verdict = run_blind_verdict(config)
    assert verdict["primary_gates"]["P2_selective_safety"] is True
    assert verdict["primary_gates"]["P3_useful_coverage"] is False
    assert verdict["primary_gates"]["P4_shift_robustness"] is True
    assert verdict["gates"]["passed"] is True  # G1-G4 all pass; P3 is separate
    assert verdict["all_primary_gates_pass"] is False
    assert verdict["result_language"] == COVERAGE_LANGUAGE


def test_contamination_failure_beats_every_other_gate(tmp_path):
    config = _config(
        tmp_path,
        _passing_cases(),
        training_texts=TRAINING_TEXTS + [_text(7)],
    )
    verdict = run_blind_verdict(config)
    assert verdict["contamination"]["checks"]["exact"]["count"] == 1
    assert verdict["contamination"]["total_flagged"] == [7]
    assert verdict["primary_gates"]["P1_data_validity"] is False
    assert verdict["result_language"] == CONTAMINATION_LANGUAGE


def test_contamination_tolerance_leaks_through_config(tmp_path):
    config = _config(
        tmp_path,
        _passing_cases(),
        training_texts=TRAINING_TEXTS + [_text(7)],
    )
    config["contamination_tolerance"] = 1
    verdict = run_blind_verdict(config)
    assert verdict["primary_gates"]["P1_data_validity"] is True
    assert verdict["all_primary_gates_pass"] is True


def test_missing_clean_slice_fails_closed(tmp_path):
    verdict = run_blind_verdict(_config(tmp_path, _shifted_slice(48, 2)))
    assert verdict["primary_gates"]["P2_selective_safety"] is False
    assert verdict["primary_gates"]["P3_useful_coverage"] is False
    assert verdict["primary_gates"]["P4_shift_robustness"] is False
    assert verdict["all_primary_gates_pass"] is False
    assert verdict["secondary"]["accuracy"] is None
    assert verdict["secondary"]["macro_f1"] is None
    assert verdict["secondary"]["bootstrap_ci"] is None
    assert verdict["sample_size_check"]["actual_accepted"] == 0
    assert verdict["sample_size_check"]["sufficient"] is False
    assert verdict["result_language"] == RISK_LANGUAGE


def test_missing_operational_measurements_fail_p6(tmp_path):
    config = _config(tmp_path, _passing_cases(), operational=None)
    verdict = run_blind_verdict(config)
    assert verdict["primary_gates"]["P6_operational_viability"] is False
    assert verdict["all_primary_gates_pass"] is False
    assert verdict["result_language"] == (
        "Operational viability requirements were not met on prospective traffic."
    )
    assert verdict["operational"]["present"] is False
    assert verdict["operational"]["failures"] == ["no operational measurements were provided"]


def test_flagged_operational_measurements_fail_p6(tmp_path):
    operational = {"latency_p95_ms": 900.0, "passed": False}
    verdict = run_blind_verdict(_config(tmp_path, _passing_cases(), operational=operational))
    assert verdict["primary_gates"]["P6_operational_viability"] is False


def test_config_disagreeing_with_the_freeze_record_is_rejected(tmp_path):
    config = _config(tmp_path, _passing_cases())
    config["epsilon"] = 0.5
    with pytest.raises(ValueError, match="does not match the frozen record"):
        run_blind_verdict(config)
    config = _config(tmp_path, _passing_cases())
    config["c_min"] = 0.5
    with pytest.raises(ValueError, match="does not match the frozen record"):
        run_blind_verdict(config)


def test_missing_config_and_case_fields_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="missing required key 'policy'"):
        run_blind_verdict({"freeze_record": {}, "cases": [], "training_texts": []})
    config = _config(tmp_path, _passing_cases())
    del config["cases"][3]["probabilities"]
    with pytest.raises(ValueError, match="case 3 is missing fields: probabilities"):
        run_blind_verdict(config)
    config = _config(tmp_path, _passing_cases())
    config["cases"][3]["candidate_count"] = 3
    with pytest.raises(ValueError, match="candidate_count"):
        run_blind_verdict(config)
    with pytest.raises(ValueError, match="at least one evaluated case"):
        run_blind_verdict(_config(tmp_path, []))


def test_perturb_text_dispatches_registered_families():
    text = "please send me the balance update"
    assert perturb_text(text, "clean", "row-1") == text
    assert perturb_text(text, "short_suffix", "row-1").startswith(text)
    assert perturb_text(text, "abbreviation", "row-1") == "pls send me the balance update"
    assert perturb_text(text, "polite_prefix", "row-1").endswith(text)
    with pytest.raises(ValueError, match="unknown slice condition"):
        perturb_text(text, "not_a_condition", "row-1")
    assert SHIFT_TAG in KNOWN_SLICE_CONDITIONS


def test_verdict_covers_the_required_protocol_outputs(tmp_path):
    """Every pre-registered output the G8 verdict must carry is present."""
    verdict = run_blind_verdict(_config(tmp_path, _passing_cases()))
    # Accepted-risk UCB and coverage on the clean slice (P2/P3 evidence).
    clean = verdict["gates"]["slices"]["clean"]
    assert clean["risk_ucb95"] is not None and clean["risk_ucb95"] <= 0.10
    assert clean["coverage"] == pytest.approx(0.96)
    # Risk-coverage curve and calibration metrics per slice.
    clean_calibration = verdict["calibration"]["slices"]["clean"]
    assert clean_calibration["risk_coverage_deciles"]
    assert isinstance(clean_calibration["ece"], float)
    assert isinstance(clean_calibration["brier"], float)
    assert isinstance(verdict["secondary"]["ece"], float)
    assert isinstance(verdict["secondary"]["brier"], float)
    # Shift-slice risk and shift coverage/retention (P4 evidence).
    shift_slice = verdict["gates"]["slices"][SHIFT_TAG]
    assert shift_slice["risk_ucb95"] is not None
    g2 = verdict["gates"]["gates"]["G2_shift_safety"]
    assert g2["per_slice"][SHIFT_TAG]["risk_ucb95"] is not None
    g3 = verdict["gates"]["gates"]["G3_coverage_retention"]
    assert g3["per_slice"][SHIFT_TAG]["coverage"] is not None
    assert g3["per_slice"][SHIFT_TAG]["retention"] >= 0.95
    # Final P1-P6 verdict, provenance, and contamination results.
    assert set(verdict["primary_gates"]) == PRIMARY_GATES
    assert verdict["all_primary_gates_pass"] is True
    assert len(verdict["freeze_record_hash"]) == 64
    assert verdict["contamination"]["format"] == "veya-contamination-report-v1"
    assert verdict["contamination"]["total_flagged"] == []


def test_contamination_precedes_statistical_evaluation(tmp_path, monkeypatch):
    """The contamination screens run before the gates and calibration report."""
    import research.veyabench.blind_verdict as blind_verdict_module

    order = []
    real_contamination = blind_verdict_module.contamination_report
    real_gates = blind_verdict_module.gate_report
    real_calibration = blind_verdict_module.calibration_report

    def contamination_first(*args, **kwargs):
        order.append("contamination")
        return real_contamination(*args, **kwargs)

    def gates_after_contamination(*args, **kwargs):
        order.append("gates")
        return real_gates(*args, **kwargs)

    def calibration_last(*args, **kwargs):
        order.append("calibration")
        return real_calibration(*args, **kwargs)

    monkeypatch.setattr(blind_verdict_module, "contamination_report", contamination_first)
    monkeypatch.setattr(blind_verdict_module, "gate_report", gates_after_contamination)
    monkeypatch.setattr(blind_verdict_module, "calibration_report", calibration_last)

    verdict = run_blind_verdict(_config(tmp_path, _passing_cases()))
    assert order == ["contamination", "gates", "calibration"]
    assert verdict["all_primary_gates_pass"] is True


def test_contamination_failure_dominates_failing_later_gates(tmp_path):
    """P1 failure decides the result language even when P2 also fails."""
    cases = _clean_slice(38, 1, 1, accepted_wrong=10) + _shifted_slice(48, 2)
    config = _config(tmp_path, cases, training_texts=TRAINING_TEXTS + [_text(7)])
    verdict = run_blind_verdict(config)
    assert verdict["primary_gates"]["P1_data_validity"] is False
    assert verdict["primary_gates"]["P2_selective_safety"] is False
    assert verdict["contamination"]["total_flagged"] == [7]
    assert verdict["result_language"] == CONTAMINATION_LANGUAGE


_CALIBRATION_PIN = {
    "path": "research/data/banking77/test.csv",
    "sha256": "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
}
_TEST_PIN = {
    "path": "data/prospective_target_domain.jsonl",
    "sha256": "a" * 64,
}


def _split_config(tmp_path, calibration, test):
    config = _config(tmp_path, _passing_cases())
    config["calibration_data"] = calibration
    config["test_data"] = test
    return config


def test_split_provenance_pins_distinct_datasets(tmp_path):
    verdict = run_blind_verdict(_split_config(tmp_path, _CALIBRATION_PIN, _TEST_PIN))
    assert verdict["split_provenance"] == {
        "calibration_data": dict(_CALIBRATION_PIN),
        "test_data": dict(_TEST_PIN),
        "disjoint": True,
    }
    assert verdict["all_primary_gates_pass"] is True
    # A legacy config without split pins reports no provenance.
    legacy = run_blind_verdict(_config(tmp_path, _passing_cases()))
    assert legacy["split_provenance"] is None


def test_split_isolation_fails_closed_on_collisions(tmp_path):
    # Same path for both sides, different hashes.
    with pytest.raises(ValueError, match="both declare path"):
        run_blind_verdict(
            _split_config(
                tmp_path,
                _CALIBRATION_PIN,
                {**_TEST_PIN, "path": _CALIBRATION_PIN["path"]},
            )
        )
    # Same content hash for both sides, different paths.
    with pytest.raises(ValueError, match="both pin content hash"):
        run_blind_verdict(
            _split_config(
                tmp_path,
                _CALIBRATION_PIN,
                {**_TEST_PIN, "sha256": _CALIBRATION_PIN["sha256"]},
            )
        )
    # Only one side declared.
    config = _config(tmp_path, _passing_cases())
    config["calibration_data"] = _CALIBRATION_PIN
    with pytest.raises(ValueError, match="test_data is missing"):
        run_blind_verdict(config)
    # Malformed entries fail closed.
    with pytest.raises(ValueError, match="sha256 must be a 64-character"):
        run_blind_verdict(
            _split_config(
                tmp_path,
                _CALIBRATION_PIN,
                {"path": "data/other.jsonl", "sha256": "not-hex"},
            )
        )
    with pytest.raises(ValueError, match="test_data must be a mapping"):
        run_blind_verdict(
            _split_config(tmp_path, _CALIBRATION_PIN, "data/other.jsonl")
        )


def test_split_isolation_precedes_contamination_screens(tmp_path, monkeypatch):
    """A split collision is rejected before the contamination screens run."""
    import research.veyabench.blind_verdict as blind_verdict_module

    def detonate(*args, **kwargs):
        raise AssertionError("contamination screens ran before split isolation")

    monkeypatch.setattr(blind_verdict_module, "contamination_report", detonate)
    config = _split_config(
        tmp_path,
        {"path": "same.jsonl", "sha256": "a" * 64},
        {"path": "same.jsonl", "sha256": "b" * 64},
    )
    with pytest.raises(ValueError, match="both declare path"):
        run_blind_verdict(config)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-xvs"]))
