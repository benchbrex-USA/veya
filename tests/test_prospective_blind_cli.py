"""End-to-end CLI verification for research/veyabench/run_prospective_blind.py.

Every dataset, freeze artifact, manifest, policy, and calibration file lives
under ``tmp_path``: the synthetic pattern of tests/test_blind_verdict.py, never
benchmark data. The CLI is invoked as a subprocess exactly the way the
workflow documents it, so the tests lock the exit-code semantics (0 PASS,
1 FAIL, 2 validation/error), the machine-readable result coverage, the split
isolation fail-closed behavior, and the placeholder blocker message.

Run: python -m pytest tests/test_prospective_blind_cli.py -xvs
"""
import json
import math
import os
import pathlib
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from research.veyabench.blind_verdict import perturb_text  # noqa: E402
from research.veyabench.freeze_record import build_freeze_record  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CLASSES = ("alpha", "beta", "delta", "epsilon", "gamma")  # sorted order
VOCAB = {label: index for index, label in enumerate(CLASSES)}
SHIFT_TAG = "short_suffix"
GIT_COMMIT = "e852981ae34990f4358979625854259302feaa78"

TRAINING_TEXTS = [
    "warehouse inventory reconciliation for the quarterly audit",
    "payroll disbursement schedule for the northern district office",
    "freight manifest consolidation for the harbor authority",
    "vaccine cold-chain inspection for the regional clinic network",
    "orchard irrigation timing for the cooperative farm board",
]

RISK_LANGUAGE = (
    "The frozen selective policy did not maintain the preregistered risk "
    "bound on prospective traffic."
)
PRIMARY_GATES = {
    "P1_data_validity",
    "P2_selective_safety",
    "P3_useful_coverage",
    "P4_shift_robustness",
    "P5_calibration_transfer",
    "P6_operational_viability",
}
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

# Four varying word slots so any two distinct indices differ in at least two
# slots; short texts with no shared boilerplate stay far below the
# contamination thresholds, so P1 passes on synthetic data by construction.
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
        _TEXT_VERBS[a], _TEXT_ADJECTIVES[b], _TEXT_AUXILIARIES[c], _TEXT_NOUNS[d], index
    )


def _record(index, tag, gold, correct, confidence, minimal=False):
    pred = gold if correct else CLASSES[(VOCAB[gold] + 1) % len(CLASSES)]
    probabilities = [0.0] * len(CLASSES)
    probabilities[VOCAB[pred]] = float(confidence)
    others = [i for i in range(len(CLASSES)) if i != VOCAB[pred]]
    share = (1.0 - confidence) / len(others)
    for i in others:
        probabilities[i] = share
    body = _text(index)
    if tag != "clean":
        body = perturb_text(body, tag, "case-%d" % index)
    record = {
        "id": "case-%04d" % index,
        "text": body,
        "label": gold,
        "scores": probabilities,
        "slice_tag": tag,
    }
    if minimal:
        # Prediction and correctness are derived: argmax(scores) over candidates.
        record["candidates"] = list(CLASSES)
    else:
        record["prediction"] = pred
        record["correct"] = bool(correct)
    return record


def _clean_records(accepted_correct, deferred_correct, deferred_wrong, accepted_wrong=0):
    records = []
    index = 0
    for _ in range(accepted_correct):
        records.append(_record(index, "clean", CLASSES[index % len(CLASSES)], True, 0.92))
        index += 1
    for _ in range(accepted_wrong):
        records.append(_record(index, "clean", CLASSES[index % len(CLASSES)], False, 0.92))
        index += 1
    for _ in range(deferred_correct):
        records.append(_record(index, "clean", CLASSES[index % len(CLASSES)], True, 0.60))
        index += 1
    for _ in range(deferred_wrong):
        records.append(_record(index, "clean", CLASSES[index % len(CLASSES)], False, 0.60))
        index += 1
    return records


def _shifted_records(accepted_correct, deferred, accepted_wrong=0):
    records = []
    index = 100
    for _ in range(accepted_correct):
        records.append(_record(index, SHIFT_TAG, CLASSES[index % len(CLASSES)], True, 0.90))
        index += 1
    for _ in range(accepted_wrong):
        records.append(_record(index, SHIFT_TAG, CLASSES[index % len(CLASSES)], False, 0.90))
        index += 1
    for _ in range(deferred):
        records.append(_record(index, SHIFT_TAG, CLASSES[index % len(CLASSES)], True, 0.55))
        index += 1
    return records


def _passing_records():
    # 48 accepted correct, 1 deferred correct, 1 deferred wrong -> clean coverage 96%.
    return _clean_records(48, 1, 1) + _shifted_records(48, 2)


def _high_risk_records():
    # 10 accepted wrong of 48 accepted clean -> risk UCB95 far above epsilon 0.10.
    return _clean_records(38, 1, 1, accepted_wrong=10) + _shifted_records(48, 2)


def _minimal_passing_records():
    """Same scenario as _passing_records with prediction/correct derived."""
    records = []
    index = 0
    for _ in range(48):
        records.append(_record(index, "clean", CLASSES[index % len(CLASSES)], True, 0.92, minimal=True))
        index += 1
    records.append(_record(index, "clean", CLASSES[index % len(CLASSES)], True, 0.60, minimal=True))
    index += 1
    records.append(_record(index, "clean", CLASSES[index % len(CLASSES)], False, 0.60, minimal=True))
    index += 1
    for _ in range(48):
        records.append(_record(index, SHIFT_TAG, CLASSES[index % len(CLASSES)], True, 0.90, minimal=True))
        index += 1
    for _ in range(2):
        records.append(_record(index, SHIFT_TAG, CLASSES[index % len(CLASSES)], True, 0.55, minimal=True))
        index += 1
    return records


def _sha256(path):
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frozen_policy():
    return {
        "format": "veya-selective-policy-v1",
        "route": "choice",
        "scorer_id": "phase12-sparse-scorer-v1",
        "score": "max_probability",
        "operating_epsilon": "disabled",
        "thresholds": {"disabled": None, "0.10": 0.85},
        "validation_split": "synthetic_validation",
        "validation_hash": "c" * 64,
        "chosen_on": "validation",
        "checkpoint_hash": "a" * 64,
        "shortlist_sha256": "b" * 64,
    }


def _build_env(base, records=None, dataset_format="jsonl", mutate=None):
    """Create a complete synthetic run environment and return its paths.

    Everything the CLI pins is created here: freeze artifacts, a release
    manifest that agrees with the freeze record, a frozen selective policy, a
    temperature artifact, a calibration dataset, the labeled dataset, and the
    run config. ``mutate(config, root, dataset)`` runs before the config is
    written so tests can tamper with pins or the dataset.
    """
    root = pathlib.Path(base) / "env"
    root.mkdir(parents=True, exist_ok=True)
    records = _passing_records() if records is None else records

    freeze = root / "freeze"
    freeze.mkdir(exist_ok=True)
    (freeze / "model.safetensors").write_bytes(b"synthetic prospective weights")
    tokenizer = freeze / "tokenizer"
    tokenizer.mkdir(exist_ok=True)
    (tokenizer / "tokenizer.json").write_text('{"version": "A"}', encoding="utf-8")
    shortlist = freeze / "shortlist_k8.pt.json"
    shortlist.write_text(json.dumps({"k": 8, "seed": 13}), encoding="utf-8")
    temperature_artifact = freeze / "calibration.json"
    temperature_artifact.write_text(
        json.dumps({"format": "veya-route-calibration-v1", "temperature": 1.0}),
        encoding="utf-8",
    )
    policy_artifact = freeze / "selective_policy.json"
    policy_artifact.write_text(json.dumps(_frozen_policy()), encoding="utf-8")

    record = build_freeze_record(
        model_path=freeze / "model.safetensors",
        tokenizer_dir=tokenizer,
        git_commit=GIT_COMMIT,
        scorer_id="phase12-sparse-scorer-v1",
        shortlist_config_path=shortlist,
        calibration_method="temperature scaling on the calibration split",
        temperature_artifact_path=temperature_artifact,
        selective_policy_algorithm="max_probability threshold at epsilon",
        selective_policy_artifact_path=policy_artifact,
        epsilon=0.10,
        c_min=0.80,
        perturbation_definitions_version="phase11-noise-v1",
        metric_implementation_version="veyabench-metrics-v1",
        exclusion_rules=["exclude exact development-set duplicates"],
        random_seeds={"bootstrap": 13},
        bootstrap_draws=1000,
        bootstrap_confidence=0.95,
        production_preprocessing_version="veya-preprocess-v1",
    )
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "format": "veya-release-manifest-v2",
                "git_commit": GIT_COMMIT,
                "base_checkpoint_hash": record["model_checkpoint_sha256"],
                "selective_policy_hash": record["selective_policy_sha256"],
                "choice": {"shortlist_hash": record["shortlist_config_sha256"]},
                "calibration": {"hierarchical_choice_hash": record["temperature_artifact_sha256"]},
                "automation": {"status": "DISABLED"},
            }
        ),
        encoding="utf-8",
    )

    calibration_data = root / "calibration.jsonl"
    calibration_data.write_text(
        "\n".join(json.dumps({"text": text}) for text in TRAINING_TEXTS) + "\n",
        encoding="utf-8",
    )

    if dataset_format == "jsonl":
        dataset = root / "dataset.jsonl"
        with open(dataset, "w", encoding="utf-8") as handle:
            for row in records:
                handle.write(json.dumps(row) + "\n")
    elif dataset_format == "json":
        dataset = root / "dataset.json"
        dataset.write_text(json.dumps(records), encoding="utf-8")
    else:
        raise ValueError("unknown dataset_format %r" % dataset_format)
    dataset_sha = _sha256(dataset)

    run_policy = dict(_frozen_policy())
    run_policy["operating_epsilon"] = "0.10"
    config = {
        "format": "veya-prospective-blind-config-v1",
        "manifest": {"path": str(manifest), "sha256": _sha256(manifest)},
        "selective_policy_artifact": {
            "path": str(policy_artifact),
            "sha256": _sha256(policy_artifact),
        },
        "temperature_artifact": {
            "path": str(temperature_artifact),
            "sha256": _sha256(temperature_artifact),
        },
        "freeze_record": record,
        "policy": run_policy,
        "calibration_data": {"path": str(calibration_data), "sha256": _sha256(calibration_data)},
        "test_data": {"path": str(dataset), "sha256": dataset_sha},
        "training_texts": list(TRAINING_TEXTS),
        "temperature": 1.0,
        "domain": "synthetic-target-domain",
        "coverage_retention_min": 0.95,
        "collapse_floor": {"coverage": 0.05, "accuracy": 0.30},
        "contamination_tolerance": 0,
        "operational": {"latency_p95_ms": 41.5, "throughput_rps": 120, "passed": True},
    }
    if mutate is not None:
        mutate(config, root, dataset)
    config_path = root / "config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config_path, dataset, {"sha256": dataset_sha, "config": config, "root": root}


def _run_cli(config_path, data_path, output_path):
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "research.veyabench.run_prospective_blind",
            "--config",
            str(config_path),
            "--data",
            str(data_path),
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=300,
    )


def test_cli_pass_run_writes_full_result_and_exits_zero(tmp_path):
    config_path, data_path, env = _build_env(tmp_path)
    out = tmp_path / "result.json"
    completed = _run_cli(config_path, data_path, out)

    assert completed.returncode == 0, completed.stderr
    lines = completed.stdout.strip().splitlines()
    assert len(lines) == 1  # exactly one line of verdict summary
    assert lines[0].startswith("PASS:")
    assert "clean coverage 96.0%" in lines[0]
    assert "UCB95 7.4%" in lines[0]

    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["format"] == "veya-prospective-blind-result-v1"
    assert result["status"] == "PASS"
    assert result["summary"] == lines[0]
    assert result["dataset"]["sha256"] == env["sha256"]
    assert result["dataset"]["path"].endswith("dataset.jsonl")
    assert result["dataset"]["n_records"] == 100
    assert result["config"]["sha256"] == _sha256(config_path)
    assert result["training_texts_count"] == len(TRAINING_TEXTS)

    verdict = result["verdict"]
    assert verdict["format"] == "veya-blind-verdict-v1"
    assert REQUIRED_VERDICT_KEYS <= set(verdict)
    assert len(verdict["freeze_record_hash"]) == 64
    assert verdict["split_provenance"] == {
        "calibration_data": env["config"]["calibration_data"],
        "test_data": env["config"]["test_data"],
        "disjoint": True,
    }
    assert set(verdict["primary_gates"]) == PRIMARY_GATES
    assert verdict["all_primary_gates_pass"] is True
    assert all(verdict["primary_gates"].values())

    clean = verdict["gates"]["slices"]["clean"]
    assert clean["risk_ucb95"] is not None and clean["risk_ucb95"] <= 0.10
    assert clean["coverage"] == pytest.approx(0.96)
    clean_calibration = verdict["calibration"]["slices"]["clean"]
    assert clean_calibration["risk_coverage_deciles"]
    assert isinstance(clean_calibration["ece"], float)
    assert isinstance(clean_calibration["brier"], float)
    assert isinstance(verdict["secondary"]["ece"], float)
    assert isinstance(verdict["secondary"]["brier"], float)
    shift_slice = verdict["gates"]["slices"][SHIFT_TAG]
    assert shift_slice["risk_ucb95"] is not None
    g2 = verdict["gates"]["gates"]["G2_shift_safety"]
    assert g2["per_slice"][SHIFT_TAG]["risk_ucb95"] is not None
    g3 = verdict["gates"]["gates"]["G3_coverage_retention"]
    assert g3["per_slice"][SHIFT_TAG]["coverage"] is not None
    assert g3["per_slice"][SHIFT_TAG]["retention"] >= 0.95
    assert verdict["contamination"]["format"] == "veya-contamination-report-v1"
    assert verdict["contamination"]["total_flagged"] == []
    assert verdict["sample_size_check"]["sufficient"] is True
    assert verdict["result_language"].startswith(
        "On prospectively collected synthetic-target-domain traffic"
    )


def test_cli_fail_run_exits_one_and_reports_the_failed_gate(tmp_path):
    config_path, data_path, _ = _build_env(tmp_path, records=_high_risk_records())
    out = tmp_path / "result.json"
    completed = _run_cli(config_path, data_path, out)

    assert completed.returncode == 1, completed.stderr
    lines = completed.stdout.strip().splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("FAIL:")
    assert "P2_selective_safety" in lines[0]

    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["status"] == "FAIL"
    verdict = result["verdict"]
    assert verdict["all_primary_gates_pass"] is False
    assert verdict["primary_gates"]["P2_selective_safety"] is False
    assert verdict["primary_gates"]["P1_data_validity"] is True
    assert verdict["result_language"] == RISK_LANGUAGE
    # A FAIL is still a full evaluation: every required metric is present.
    assert REQUIRED_VERDICT_KEYS <= set(verdict)
    assert verdict["gates"]["slices"]["clean"]["risk_ucb95"] > 0.10
    assert verdict["calibration"]["slices"]["clean"]["risk_coverage_deciles"]


def test_cli_accepts_a_json_array_dataset(tmp_path):
    config_path, data_path, _ = _build_env(tmp_path, dataset_format="json")
    out = tmp_path / "result.json"
    completed = _run_cli(config_path, data_path, out)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["status"] == "PASS"
    assert result["dataset"]["n_records"] == 100
    assert result["dataset"]["path"].endswith("dataset.json")


def test_cli_derives_prediction_and_correct_from_scores(tmp_path):
    """Records carrying only candidates+scores produce the identical verdict."""
    config_a, data_a, _ = _build_env(tmp_path / "explicit")
    config_b, data_b, _ = _build_env(tmp_path / "derived", records=_minimal_passing_records())
    out_a = tmp_path / "result-explicit.json"
    out_b = tmp_path / "result-derived.json"

    first = _run_cli(config_a, data_a, out_a)
    second = _run_cli(config_b, data_b, out_b)
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr

    verdict_a = json.loads(out_a.read_text(encoding="utf-8"))["verdict"]
    verdict_b = json.loads(out_b.read_text(encoding="utf-8"))["verdict"]
    # split_provenance pins each run's own dataset file, so it legitimately
    # differs; every statistic and gate must be identical.
    verdict_a.pop("split_provenance")
    verdict_b.pop("split_provenance")
    assert verdict_b == verdict_a


def test_placeholder_test_data_fails_closed_with_the_blocker_message(tmp_path):
    def make_placeholder(config, root, dataset):
        config["test_data"] = {
            "path": "data/target_domain/REPLACE_WITH_TARGET_DOMAIN_LABELED_DATASET.jsonl",
            "sha256": "0" * 64,
        }

    config_path, data_path, _ = _build_env(tmp_path, mutate=make_placeholder)
    out = tmp_path / "result.json"
    completed = _run_cli(config_path, data_path, out)

    assert completed.returncode == 2
    assert completed.stderr.startswith("error:")
    assert "supply a target-domain labeled dataset" in completed.stderr
    assert "G6/G8" in completed.stderr
    assert not out.exists()


def test_split_hash_collision_fails_closed(tmp_path):
    """A calibration copy of the dataset shares its hash: exit 2, no result."""

    def collide_hashes(config, root, dataset):
        copy = root / "dataset_copy.jsonl"
        shutil.copyfile(dataset, copy)
        config["calibration_data"] = {
            "path": str(copy),
            "sha256": config["test_data"]["sha256"],
        }

    config_path, data_path, _ = _build_env(tmp_path, mutate=collide_hashes)
    out = tmp_path / "result.json"
    completed = _run_cli(config_path, data_path, out)

    assert completed.returncode == 2
    assert "must be distinct datasets" in completed.stderr
    assert "both pin content hash" in completed.stderr
    assert not out.exists()


def test_split_path_collision_fails_closed(tmp_path):
    def collide_paths(config, root, dataset):
        config["calibration_data"] = dict(config["test_data"])

    config_path, data_path, _ = _build_env(tmp_path, mutate=collide_paths)
    out = tmp_path / "result.json"
    completed = _run_cli(config_path, data_path, out)

    assert completed.returncode == 2
    assert "must be distinct datasets" in completed.stderr
    assert "both declare path" in completed.stderr
    assert not out.exists()


def test_second_run_refuses_to_overwrite_the_result(tmp_path):
    config_path, data_path, _ = _build_env(tmp_path)
    out = tmp_path / "result.json"

    first = _run_cli(config_path, data_path, out)
    assert first.returncode == 0, first.stderr
    original = out.read_text(encoding="utf-8")

    second = _run_cli(config_path, data_path, out)
    assert second.returncode == 2
    assert "already exists" in second.stderr
    assert out.read_text(encoding="utf-8") == original


# ---------------------------------------------------------------------------
# Validation errors: exit 2, one-line error on stderr, no result written.
# ---------------------------------------------------------------------------

def _case_dataset_tampered_after_pinning(tmp_path):
    def tamper(config, root, dataset):
        with open(dataset, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(_record(199, "clean", "alpha", True, 0.92)) + "\n")

    config_path, data_path, _ = _build_env(tmp_path, mutate=tamper)
    return config_path, data_path, "test_data.sha256"


def _case_scores_do_not_sum_to_one(tmp_path):
    records = _passing_records()
    broken = dict(records[0])
    broken["scores"] = [0.5, 0.05, 0.05, 0.05, 0.05]
    config_path, data_path, _ = _build_env(tmp_path, records=[broken] + records[1:])
    return config_path, data_path, "must sum to 1.0"


def _case_inconsistent_correct_flag(tmp_path):
    records = _passing_records()
    bad = dict(records[0])
    bad["correct"] = False  # prediction == label here, so correct must be True
    config_path, data_path, _ = _build_env(tmp_path, records=[bad] + records[1:])
    return config_path, data_path, "contradicts prediction"


def _case_unknown_record_field(tmp_path):
    records = _passing_records()
    bad = dict(records[0])
    bad["scoers"] = [0.9, 0.1, 0.0, 0.0, 0.0]
    config_path, data_path, _ = _build_env(tmp_path, records=[bad] + records[1:])
    return config_path, data_path, "unknown field"


def _case_tampered_policy_thresholds(tmp_path):
    def tamper(config, root, dataset):
        config["policy"] = dict(config["policy"])
        config["policy"]["thresholds"] = {"disabled": None, "0.10": 0.50}

    config_path, data_path, _ = _build_env(tmp_path, mutate=tamper)
    return config_path, data_path, "differs from the frozen selective policy"


def _case_temperature_disagrees_with_artifact(tmp_path):
    def tamper(config, root, dataset):
        config["temperature"] = 1.5

    config_path, data_path, _ = _build_env(tmp_path, mutate=tamper)
    return config_path, data_path, "frozen temperature artifact"


def _case_missing_temperature(tmp_path):
    def tamper(config, root, dataset):
        config.pop("temperature")

    config_path, data_path, _ = _build_env(tmp_path, mutate=tamper)
    return config_path, data_path, "missing required key"


def _case_both_training_text_sources(tmp_path):
    def tamper(config, root, dataset):
        config["training_texts_path"] = str(root / "calibration.jsonl")

    config_path, data_path, _ = _build_env(tmp_path, mutate=tamper)
    return config_path, data_path, "exactly one of training_texts"


def _case_calibration_hash_mismatch(tmp_path):
    def tamper(config, root, dataset):
        config["calibration_data"] = {"path": config["calibration_data"]["path"], "sha256": "d" * 64}

    config_path, data_path, _ = _build_env(tmp_path, mutate=tamper)
    return config_path, data_path, "calibration_data hash mismatch"


def _case_drifted_manifest(tmp_path):
    def tamper(config, root, dataset):
        config["manifest"] = {"path": config["manifest"]["path"], "sha256": "e" * 64}

    config_path, data_path, _ = _build_env(tmp_path, mutate=tamper)
    return config_path, data_path, "manifest hash mismatch"


_ERROR_CASES = (
    ("dataset_hash_mismatch", _case_dataset_tampered_after_pinning),
    ("scores_do_not_sum", _case_scores_do_not_sum_to_one),
    ("inconsistent_correct_flag", _case_inconsistent_correct_flag),
    ("unknown_record_field", _case_unknown_record_field),
    ("tampered_policy_thresholds", _case_tampered_policy_thresholds),
    ("temperature_mismatch", _case_temperature_disagrees_with_artifact),
    ("missing_temperature", _case_missing_temperature),
    ("both_training_text_sources", _case_both_training_text_sources),
    ("calibration_hash_mismatch", _case_calibration_hash_mismatch),
    ("drifted_manifest", _case_drifted_manifest),
)


@pytest.mark.parametrize(
    "name,factory", _ERROR_CASES, ids=[name for name, _factory in _ERROR_CASES]
)
def test_cli_validation_errors_fail_closed_with_exit_two(tmp_path, name, factory):
    config_path, data_path, expected_snippet = factory(tmp_path)
    out = tmp_path / "result.json"

    completed = _run_cli(config_path, data_path, out)

    assert completed.returncode == 2, completed.stdout + completed.stderr
    assert completed.stderr.startswith("error:")
    assert expected_snippet in completed.stderr
    assert completed.stdout == ""
    assert not out.exists()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-xvs"]))
