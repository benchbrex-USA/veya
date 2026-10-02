"""Release metadata assembler (G22): schema, accuracy, drift, and roundtrip.

Reads the checked-in release artifacts under ``artifacts/release/`` only:
no checkpoint, tokenizer, network, or write outside ``tmp_path``.
"""
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from research.veyabench.release_metadata import (  # noqa: E402
    ARTIFACTS,
    CORE_ARTIFACTS,
    METADATA_FORMAT,
    REPO_ROOT,
    assemble_release_metadata,
)

HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")

REQUIRED_TOP_LEVEL = {
    "format",
    "generated",
    "source",
    "dependencies",
    "model_artifacts",
    "drift",
    "calibration",
    "config",
    "ci",
    "build",
    "evaluation_references",
    "production_scope",
}

SECTIONS = tuple(sorted(REQUIRED_TOP_LEVEL - {"format", "generated"}))


def load_release(name):
    return json.loads((ARTIFACTS / name).read_text(encoding="utf-8"))


def recompute(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


@pytest.fixture(scope="module")
def metadata():
    return assemble_release_metadata()


def test_required_top_level_keys(metadata):
    assert isinstance(metadata, dict)
    assert REQUIRED_TOP_LEVEL <= set(metadata)
    for section in SECTIONS:
        assert isinstance(metadata[section], dict)


def test_format_identifier(metadata):
    assert metadata["format"] == METADATA_FORMAT == "veya-release-metadata-v1"


def test_git_commit_is_a_sha_or_unknown(metadata):
    commit = metadata["source"]["git_commit"]
    assert isinstance(commit, str)
    assert commit == "unknown" or HEX40.fullmatch(commit)


def test_lock_hash_is_sha256_or_missing(metadata):
    lock_hash = metadata["dependencies"]["lock_hash"]
    assert isinstance(lock_hash, str)
    assert lock_hash == "missing" or HEX64.fullmatch(lock_hash)


def test_manifest_hash_is_present(metadata):
    artifacts = metadata["model_artifacts"]
    assert HEX64.fullmatch(artifacts["manifest_hash"])
    assert HEX64.fullmatch(artifacts["policy_hash"])
    assert HEX64.fullmatch(metadata["dependencies"]["sbom_hash"])


def test_ci_fields_always_have_values(metadata, monkeypatch):
    for field in ("pipeline_run_id", "pipeline_url", "triggered_by", "ref"):
        value = metadata["ci"][field]
        assert isinstance(value, str) and value, "ci.%s must always carry a value" % field

    for var in (
        "GITHUB_RUN_ID",
        "GITHUB_SERVER_URL",
        "GITHUB_REPOSITORY",
        "GITHUB_ACTOR",
        "GITHUB_REF",
    ):
        monkeypatch.delenv(var, raising=False)
    local = assemble_release_metadata()["ci"]
    assert local == {
        "pipeline_run_id": "local",
        "pipeline_url": "local",
        "triggered_by": "local",
        "ref": "local",
    }

    monkeypatch.setenv("GITHUB_RUN_ID", "42")
    monkeypatch.setenv("GITHUB_SERVER_URL", "https://github.com")
    monkeypatch.setenv("GITHUB_REPOSITORY", "veyadynamics/veya")
    monkeypatch.setenv("GITHUB_ACTOR", "benchbrex")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    ci = assemble_release_metadata()["ci"]
    assert ci["pipeline_run_id"] == "42"
    assert ci["pipeline_url"] == "https://github.com/veyadynamics/veya/actions/runs/42"
    assert ci["triggered_by"] == "benchbrex"
    assert ci["ref"] == "refs/heads/main"


def test_json_roundtrip(metadata, tmp_path):
    out = tmp_path / "release_metadata.json"
    out.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    assert json.loads(out.read_text(encoding="utf-8")) == metadata


def test_generated_timestamp_is_iso8601(metadata):
    for value in (metadata["generated"], metadata["build"]["timestamp"]):
        assert isinstance(value, str) and value.endswith("Z")
        parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
        assert parsed.tzinfo is not None
        assert parsed.utcoffset() == datetime.timedelta(0)


def test_metadata_mirrors_manifest_and_policy(metadata):
    """Calibration/config/model sections quote the source documents verbatim."""
    manifest = load_release("manifest.json")
    policy = load_release("selective_policy.json")

    assert metadata["calibration"] == manifest["calibration"]
    assert metadata["config"] == {
        "scorer_id": policy["scorer_id"],
        "score_method": policy["score"],
        "route": policy["route"],
        "operating_epsilon": policy["operating_epsilon"],
        "architecture_version": manifest["architecture_version"],
    }

    artifacts = metadata["model_artifacts"]
    assert artifacts["manifest_hash"] == recompute(ARTIFACTS / "manifest.json")
    assert artifacts["policy_hash"] == recompute(ARTIFACTS / "selective_policy.json")
    assert artifacts["release"] == manifest["release"]
    assert artifacts["checkpoint_path"] == manifest["base_checkpoint"]
    assert artifacts["checkpoint_hash"] == manifest["base_checkpoint_hash"]
    assert artifacts["shortlist_hash"] == policy["shortlist_sha256"]


def test_evaluation_references_cover_release_artifacts(metadata):
    expected = {
        "artifacts/release/%s" % path.name
        for path in ARTIFACTS.glob("*.json")
        if path.name not in CORE_ARTIFACTS
    }
    references = metadata["evaluation_references"]
    assert set(references) == expected
    assert "artifacts/release/blind_clinc150_val_result.json" in references
    for path, digest in references.items():
        assert HEX64.fullmatch(digest), path
        assert digest == recompute(REPO_ROOT / path)


def test_runtime_drift_matches_recomputed_hashes(metadata):
    """Drift lists and current hashes must agree with the working tree."""
    manifest = load_release("manifest.json")
    drift = metadata["drift"]
    mismatched, missing = [], []
    for rel_path, pinned in manifest["runtime_files"].items():
        path = REPO_ROOT / rel_path
        if not path.is_file():
            missing.append(rel_path)
        elif recompute(path) != pinned:
            mismatched.append(rel_path)
    assert drift["runtime_files_checked"] == len(manifest["runtime_files"])
    assert drift["runtime_files_mismatched"] == sorted(mismatched)
    assert drift["runtime_files_missing"] == sorted(missing)
    assert drift["runtime_file_hashes"] == {
        rel_path: recompute(REPO_ROOT / rel_path)
        for rel_path in sorted(manifest["runtime_files"])
        if (REPO_ROOT / rel_path).is_file()
    }
    assert isinstance(drift["note"], str) and drift["note"]


def test_cli_writes_output(tmp_path):
    out = tmp_path / "release_metadata.json"
    result = subprocess.run(
        [sys.executable, "-m", "research.veyabench.release_metadata", "--output", str(out)],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert ("Release metadata written to %s" % out) in result.stdout
    assert json.loads(out.read_text(encoding="utf-8"))["format"] == METADATA_FORMAT
