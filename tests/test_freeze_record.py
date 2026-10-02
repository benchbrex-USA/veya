"""Freeze-record assembly, canonical hashing, and format validation.

All inputs are synthetic files under ``tmp_path``: no checkpoint, tokenizer,
or git state outside the test sandbox is touched.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hashlib
import json
import pathlib

import pytest

from research.veyabench.freeze_record import (  # noqa: E402
    FREEZE_RECORD_FORMAT,
    build_freeze_record,
    freeze_record_hash,
    validate_freeze_record,
)

EXPECTED_KEYS = {
    "format",
    "model_checkpoint_sha256",
    "tokenizer_version_sha256",
    "inference_code_commit",
    "scorer_id",
    "shortlist_config_sha256",
    "calibration_method",
    "temperature_artifact_sha256",
    "selective_policy_algorithm",
    "selective_policy_sha256",
    "epsilon",
    "c_min",
    "perturbation_definitions_version",
    "metric_implementation_version",
    "exclusion_rules",
    "random_seeds",
    "bootstrap_draws",
    "bootstrap_confidence",
    "production_preprocessing_version",
}

HASH_FIELDS = (
    "model_checkpoint_sha256",
    "tokenizer_version_sha256",
    "shortlist_config_sha256",
    "temperature_artifact_sha256",
    "selective_policy_sha256",
)


def make_env(root):
    """Create a synthetic freeze environment; return (kwargs, files)."""
    root = pathlib.Path(root)
    root.mkdir(parents=True, exist_ok=True)
    model = root / "model.safetensors"
    model.write_bytes(b"synthetic weights payload")
    tokenizer = root / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text('{"version": "A"}', encoding="utf-8")
    (tokenizer / "tokenizer_config.json").write_text(
        '{"model_max_length": 512}', encoding="utf-8"
    )
    shortlist = root / "shortlist_k8.pt.json"
    shortlist.write_text(json.dumps({"k": 8, "seed": 13}), encoding="utf-8")
    temperature = root / "calibration.json"
    temperature.write_text(
        json.dumps({"method": "temperature", "T": 1.25}), encoding="utf-8"
    )
    policy = root / "selective_policy.json"
    policy.write_text(json.dumps({"score": "max_probability"}), encoding="utf-8")
    kwargs = dict(
        model_path=model,
        tokenizer_dir=tokenizer,
        git_commit="e852981ae34990f4358979625854259302feaa78",
        scorer_id="phase12-sparse-scorer-v1",
        shortlist_config_path=shortlist,
        calibration_method="temperature scaling on the validation split",
        temperature_artifact_path=temperature,
        selective_policy_algorithm="max_probability threshold at epsilon",
        selective_policy_artifact_path=policy,
        epsilon=0.02,
        c_min=0.6,
        perturbation_definitions_version="phase11-noise-v3",
        metric_implementation_version="veyabench-metrics-v2",
        exclusion_rules=[
            "exclude exact development-set duplicates",
            "exclude languages \u2209 Latin script",
        ],
        random_seeds={"bootstrap": 13, "shuffle": 20260929},
        bootstrap_draws=10000,
        bootstrap_confidence=0.95,
        production_preprocessing_version="veya-preprocess-v1",
    )
    files = {
        "model": model,
        "tokenizer": tokenizer,
        "shortlist": shortlist,
        "temperature": temperature,
        "policy": policy,
    }
    return kwargs, files


def _mutations():
    """One mutation per Section 2 field: (name, callable(kwargs, files))."""
    return [
        (
            "model_bytes",
            lambda kw, env: env["model"].write_bytes(b"synthetic weights payload v2"),
        ),
        (
            "tokenizer_bytes",
            lambda kw, env: (env["tokenizer"] / "tokenizer.json").write_text(
                '{"version": "B"}', encoding="utf-8"
            ),
        ),
        (
            "tokenizer_extra_file",
            lambda kw, env: (env["tokenizer"] / "special_tokens_map.json").write_text(
                "{}", encoding="utf-8"
            ),
        ),
        (
            "tokenizer_renamed_file",
            lambda kw, env: (env["tokenizer"] / "tokenizer_config.json").rename(
                env["tokenizer"] / "tokenizer_cfg.json"
            ),
        ),
        (
            "git_commit",
            lambda kw, env: kw.update(
                git_commit="0123456789abcdef0123456789abcdef01234567"
            ),
        ),
        ("scorer_id", lambda kw, env: kw.update(scorer_id="phase12-sparse-scorer-v2")),
        (
            "shortlist_config",
            lambda kw, env: env["shortlist"].write_text(
                json.dumps({"k": 16, "seed": 13}), encoding="utf-8"
            ),
        ),
        (
            "calibration_method",
            lambda kw, env: kw.update(calibration_method="fixed T=1.0"),
        ),
        (
            "temperature_artifact",
            lambda kw, env: env["temperature"].write_text(
                json.dumps({"method": "temperature", "T": 1.5}), encoding="utf-8"
            ),
        ),
        (
            "selective_policy_algorithm",
            lambda kw, env: kw.update(
                selective_policy_algorithm="margin threshold at epsilon"
            ),
        ),
        (
            "selective_policy_artifact",
            lambda kw, env: env["policy"].write_text(
                json.dumps({"score": "margin"}), encoding="utf-8"
            ),
        ),
        ("epsilon", lambda kw, env: kw.update(epsilon=0.01)),
        ("c_min", lambda kw, env: kw.update(c_min=0.5)),
        (
            "perturbation_definitions_version",
            lambda kw, env: kw.update(
                perturbation_definitions_version="phase11-noise-v4"
            ),
        ),
        (
            "metric_implementation_version",
            lambda kw, env: kw.update(
                metric_implementation_version="veyabench-metrics-v3"
            ),
        ),
        (
            "exclusion_rules",
            lambda kw, env: kw.update(
                exclusion_rules=[*kw["exclusion_rules"], "exclude duplicate ids"]
            ),
        ),
        (
            "random_seeds",
            lambda kw, env: kw.update(
                random_seeds={**kw["random_seeds"], "bootstrap": 99}
            ),
        ),
        ("bootstrap_draws", lambda kw, env: kw.update(bootstrap_draws=20000)),
        (
            "bootstrap_confidence",
            lambda kw, env: kw.update(bootstrap_confidence=0.99),
        ),
        (
            "production_preprocessing_version",
            lambda kw, env: kw.update(
                production_preprocessing_version="veya-preprocess-v2"
            ),
        ),
    ]


_MUTATIONS = _mutations()


def test_record_pins_every_section2_field(tmp_path):
    kwargs, _files = make_env(tmp_path / "env")
    record = build_freeze_record(**kwargs)
    assert set(record) == EXPECTED_KEYS
    assert record["format"] == FREEZE_RECORD_FORMAT == "veya-freeze-record-v1"
    assert record["epsilon"] == 0.02
    assert record["c_min"] == 0.6
    assert record["exclusion_rules"] == kwargs["exclusion_rules"]
    assert record["random_seeds"] == kwargs["random_seeds"]
    assert record["bootstrap_draws"] == 10000
    assert record["bootstrap_confidence"] == 0.95
    assert record["model_checkpoint_sha256"] == hashlib.sha256(
        b"synthetic weights payload"
    ).hexdigest()
    for field in HASH_FIELDS:
        assert len(record[field]) == 64
        int(record[field], 16)  # parses as 64-digit lowercase hex
    validate_freeze_record(record)


def test_same_inputs_produce_the_same_hash(tmp_path):
    kwargs_a, _files_a = make_env(tmp_path / "env_a")
    kwargs_b, _files_b = make_env(tmp_path / "env_b")
    record_a = build_freeze_record(**kwargs_a)
    record_b = build_freeze_record(**kwargs_b)
    assert record_a == record_b  # content-identical inputs in different directories
    hash_a = freeze_record_hash(record_a)
    assert hash_a == freeze_record_hash(record_b)
    assert freeze_record_hash(build_freeze_record(**kwargs_a)) == hash_a
    assert len(hash_a) == 64
    int(hash_a, 16)


def test_hash_is_sha256_of_the_canonical_json_serialization(tmp_path):
    kwargs, _files = make_env(tmp_path / "env")
    record = build_freeze_record(**kwargs)
    canonical = json.dumps(
        record, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    assert freeze_record_hash(record) == hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()
    # The record carries non-ASCII text, so this also pins ensure_ascii=False.
    assert any(any(ord(ch) > 127 for ch in rule) for rule in record["exclusion_rules"])


@pytest.mark.parametrize(
    ("name", "mutate"),
    _MUTATIONS,
    ids=[name for name, _mutate in _MUTATIONS],
)
def test_changing_any_field_changes_the_hash(tmp_path, name, mutate):
    base_kwargs, _base_files = make_env(tmp_path / "base")
    base_hash = freeze_record_hash(build_freeze_record(**base_kwargs))
    kwargs, files = make_env(tmp_path / "mutated")
    assert freeze_record_hash(build_freeze_record(**kwargs)) == base_hash
    mutate(kwargs, files)
    assert freeze_record_hash(build_freeze_record(**kwargs)) != base_hash


def test_format_validation_rejects_tampered_records(tmp_path):
    kwargs, _files = make_env(tmp_path / "env")
    record = build_freeze_record(**kwargs)

    wrong_format = dict(record, format="veya-freeze-record-v2")
    with pytest.raises(ValueError, match="format"):
        validate_freeze_record(wrong_format)
    with pytest.raises(ValueError, match="format"):
        freeze_record_hash(wrong_format)

    no_format = {key: value for key, value in record.items() if key != "format"}
    with pytest.raises(ValueError, match="format"):
        freeze_record_hash(no_format)

    missing_field = {key: value for key, value in record.items() if key != "epsilon"}
    with pytest.raises(ValueError, match="epsilon"):
        validate_freeze_record(missing_field)

    bad_hash = dict(record, model_checkpoint_sha256="deadbeef")
    with pytest.raises(ValueError, match="model_checkpoint_sha256"):
        validate_freeze_record(bad_hash)

    bad_scalar = dict(record, epsilon=1.5)
    with pytest.raises(ValueError, match="epsilon"):
        validate_freeze_record(bad_scalar)

    with pytest.raises(ValueError, match="JSON object"):
        validate_freeze_record(["not", "a", "record"])


def test_build_rejects_invalid_inputs(tmp_path):
    kwargs, _files = make_env(tmp_path / "env")
    cases = (
        ({"model_path": tmp_path / "missing.safetensors"}, "model_path"),
        ({"shortlist_config_path": tmp_path / "missing.json"}, "shortlist_config_path"),
        ({"tokenizer_dir": tmp_path / "missing_tokenizer"}, "not found"),
        ({"epsilon": 0.0}, "epsilon"),
        ({"epsilon": 1.0}, "epsilon"),
        ({"c_min": 0.0}, "c_min"),
        ({"c_min": 1.5}, "c_min"),
        ({"bootstrap_confidence": 0.0}, "bootstrap_confidence"),
        ({"bootstrap_confidence": 1.0}, "bootstrap_confidence"),
        ({"bootstrap_draws": 0}, "bootstrap_draws"),
        ({"bootstrap_draws": 10000.5}, "bootstrap_draws"),
        ({"git_commit": ""}, "git_commit"),
        ({"exclusion_rules": "no overlaps"}, "exclusion_rules"),
        ({"exclusion_rules": ["ok", ""]}, "exclusion_rules"),
        ({"random_seeds": {"bootstrap": 13.5}}, "random_seeds"),
        ({"random_seeds": {"bootstrap": True}}, "random_seeds"),
    )
    for override, message in cases:
        with pytest.raises(ValueError, match=message):
            build_freeze_record(**{**kwargs, **override})


def test_empty_tokenizer_directory_is_rejected(tmp_path):
    kwargs, _files = make_env(tmp_path / "env")
    empty = tmp_path / "empty_tokenizer"
    empty.mkdir()
    with pytest.raises(ValueError, match="no files"):
        build_freeze_record(**{**kwargs, "tokenizer_dir": empty})
