"""Tests for production artifact integrity validation (G14).

Every SHA-256 pin recorded in artifacts/release/manifest.json is recomputed
against the working tree: served model artifacts, release artifacts, audit
inputs, and the hash-pinned runtime module set. Environment artifacts built
for the same gate (sbom.json, requirements.lock) are validated structurally.
The checkpoint itself is gitignored, so its hash test skips when the weights
are not present locally.
"""
import hashlib
import json
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "artifacts" / "release"
MODELS_DIR = ROOT / "models" / "production"

_ISO_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_SBOM_CATEGORIES = {"runtime", "train", "dev", "self"}
_CORE_DEPENDENCIES = {"torch", "transformers", "safetensors", "huggingface-hub", "numpy"}


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _load(name: str) -> dict:
    return json.loads((ARTIFACTS_DIR / name).read_text(encoding="utf-8"))


@pytest.fixture
def manifest():
    p = ARTIFACTS_DIR / "manifest.json"
    if not p.exists():
        pytest.skip("manifest.json not found")
    return json.loads(p.read_text(encoding="utf-8"))


def _check_hashes(cases):
    """Assert every (label, repo-relative path, sha256) pin holds.

    Tracked artifacts must be present: a missing one is an integrity failure.
    Mismatches and missing files are reported together for one clear signal.
    """
    missing = []
    mismatched = []
    for label, rel_path, expected in cases:
        path = ROOT / rel_path
        if not path.is_file():
            missing.append("%s (%s)" % (label, rel_path))
            continue
        actual = _sha256(path)
        if actual != expected:
            mismatched.append("%s (%s): %s != %s" % (label, rel_path, actual[:16], expected[:16]))
    problems = []
    if mismatched:
        problems.append("hash mismatches:\n  " + "\n  ".join(mismatched))
    if missing:
        problems.append("missing artifacts:\n  " + "\n  ".join(missing))
    assert not problems, "\n".join(problems)


class TestArtifactIntegrity:
    def test_manifest_exists(self):
        assert (ARTIFACTS_DIR / "manifest.json").exists()

    def test_manifest_format(self, manifest):
        assert manifest["format"] == "veya-release-manifest-v2"
        assert manifest.get("release")
        assert manifest.get("base_checkpoint_hash")
        assert (ROOT / manifest["base_checkpoint"]).is_dir()

    def test_selective_policy_exists(self):
        assert (ARTIFACTS_DIR / "selective_policy.json").exists()

    def test_selective_policy_integrity(self):
        """Verify selective policy is valid JSON with required fields."""
        policy = _load("selective_policy.json")
        assert policy["format"] == "veya-selective-policy-v1"
        assert "thresholds" in policy
        assert "checkpoint_hash" in policy
        assert policy["operating_epsilon"] in policy["thresholds"]

    def test_selective_policy_bindings(self, manifest):
        """The policy must belong to the exact checkpoint, shortlist, and route."""
        policy = _load("selective_policy.json")
        assert policy["checkpoint_hash"] == manifest["base_checkpoint_hash"]
        assert policy["shortlist_sha256"] == manifest["choice"]["shortlist_hash"]
        assert policy["route"] == manifest["calibration"]["hierarchical_route"]
        assert policy["scorer_id"] == manifest["served_scorer_id"]
        assert policy["chosen_on"] == "validation"
        assert policy.get("validation_hash")

    def test_sbom_exists(self):
        assert (ARTIFACTS_DIR / "sbom.json").exists()

    def test_sbom_valid_structure(self):
        sbom = _load("sbom.json")
        assert sbom["format"] == "veya-sbom-v1"
        assert _ISO_UTC.match(sbom["generated"])
        assert sbom["python_version"].startswith("3.")
        assert "packages" in sbom
        assert len(sbom["packages"]) > 0
        names = []
        for pkg in sbom["packages"]:
            assert "name" in pkg and pkg["name"]
            assert "version" in pkg and pkg["version"]
            assert pkg["license"], "license missing for %s" % pkg["name"]
            assert pkg["source"].startswith("http")
            assert pkg["category"] in _SBOM_CATEGORIES
            assert isinstance(pkg["direct"], bool)
            names.append(pkg["name"].lower().replace("_", "-"))
        assert len(names) == len(set(names)), "duplicate packages in SBOM"
        assert "torch" in names and "numpy" in names
        assert any(pkg["category"] == "runtime" for pkg in sbom["packages"])

    def test_requirements_lock_exists(self):
        lock = ROOT / "requirements.lock"
        assert lock.exists(), "requirements.lock not found"

    def test_requirements_lock_pinned_and_covers_declared_deps(self):
        pins = [
            line.strip()
            for line in (ROOT / "requirements.lock").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        assert len(pins) >= 20, "lock file is suspiciously small: %d pins" % len(pins)
        names = []
        for pin in pins:
            assert "==" in pin, "unpinned requirement: %r" % pin
            name = pin.split("==", 1)[0].split("[")[0].strip().lower().replace("_", "-")
            assert name, "unparsable pin: %r" % pin
            names.append(name)
        assert len(names) == len(set(names)), "duplicate pins in lock file"
        missing = _CORE_DEPENDENCIES - set(names)
        assert not missing, "lock file misses declared dependencies: %s" % sorted(missing)

    def test_checkpoint_hash_matches_manifest(self, manifest):
        """Verify the main checkpoint hash matches manifest."""
        checkpoint = MODELS_DIR / "model.safetensors"
        if not checkpoint.exists():
            pytest.skip("checkpoint not present locally")
        expected = (
            manifest.get("base_checkpoint_hash")
            or manifest.get("checkpoint_hash")
            or manifest.get("checkpoint_sha256")
        )
        if not expected:
            pytest.skip("no checkpoint hash in manifest")
        assert _sha256(checkpoint) == expected

    def test_model_artifact_hashes(self, manifest):
        """Config, tokenizer, router, shortlist, and calibration pins hold."""
        cases = [
            ("model_config", manifest["model_config"], manifest["model_config_hash"]),
            ("choice_router", manifest["routing"]["choice_router"], manifest["routing"]["choice_router_hash"]),
            ("shortlist", manifest["choice"]["shortlist_artifact"], manifest["choice"]["shortlist_hash"]),
            ("packed_calibration", manifest["calibration"]["packed_choice"], manifest["calibration"]["packed_choice_hash"]),
            (
                "route_calibration",
                manifest["calibration"]["hierarchical_choice"],
                manifest["calibration"]["hierarchical_choice_hash"],
            ),
        ]
        for rel_path, expected in manifest["tokenizer_files"].items():
            cases.append(("tokenizer %s" % rel_path, rel_path, expected))
        _check_hashes(cases)

    def test_release_artifact_hashes(self, manifest):
        """Selective policy, cutoff evidence, and the blind result match."""
        _check_hashes([
            ("selective_policy", manifest["selective_policy"], manifest["selective_policy_hash"]),
            ("route_cutoff_audit", manifest["routing"]["cutoff_evidence"], manifest["routing"]["cutoff_evidence_hash"]),
            ("blind_result", manifest["blind_evaluation"]["result"], manifest["blind_evaluation"]["result_hash"]),
        ])

    def test_audit_input_hashes(self, manifest):
        """Data lineage, freeze record, blind dataset, and harness match."""
        blind = manifest["blind_evaluation"]
        _check_hashes([
            ("data_lineage", manifest["data_lineage"], manifest["data_lineage_hash"]),
            ("freeze_manifest", blind["freeze_manifest"], blind["freeze_manifest_hash"]),
            ("blind_dataset", blind["dataset"], blind["dataset_hash"]),
            ("blind_harness", blind["harness"], blind["harness_hash"]),
        ])

    def test_research_artifact_hashes(self, manifest):
        """The not-served phase-9 scorer artifacts still match their pins."""
        phase9 = manifest["research_artifacts_not_served"]["phase9_hypothesis"]
        _check_hashes([
            ("phase9_scorer_head", phase9["scorer_head"], phase9["scorer_head_hash"]),
            ("phase9_encoder_delta", phase9["encoder_delta"], phase9["encoder_delta_hash"]),
        ])

    def test_runtime_files_match_manifest(self, manifest):
        """The hash-pinned runtime modules must match the manifest exactly."""
        runtime = manifest.get("runtime_files") or {}
        assert runtime, "manifest lists no runtime files"
        missing = []
        mismatched = []
        for rel_path, expected in runtime.items():
            path = ROOT / rel_path
            if not path.is_file():
                missing.append(rel_path)
                continue
            if _sha256(path) != expected:
                mismatched.append(rel_path)
        assert not missing, "pinned runtime files are missing: %s" % sorted(missing)
        assert not mismatched, "runtime files changed after the manifest was minted: %s" % sorted(mismatched)

    def test_runtime_audit_exists(self):
        assert (ARTIFACTS_DIR / "runtime_audit.json").exists()
        audit = _load("runtime_audit.json")
        assert audit["format"] == "veya-runtime-audit-v1"

    def test_runtime_audit_matches_manifest(self):
        """The runtime audit must have been run against this exact manifest."""
        audit = _load("runtime_audit.json")
        assert audit["manifest_sha256"] == _sha256(ARTIFACTS_DIR / "manifest.json")
        assert audit["all_version_fields_match"] is True

    def test_json_artifacts_parse(self):
        """Every release artifact under artifacts/release/ parses as JSON."""
        sweep = [
            "manifest.json",
            "manifest-phase5.json",
            "manifest-phase9.json",
            "phase_audit.json",
            "production_scope.json",
            "system_map.json",
            "route_cutoff_audit.json",
            "runtime_audit.json",
            "selective_policy.json",
            "sbom.json",
            "blind_clinc150_val_result.json",
            "blind_clinc150_test_failed_run.json",
        ]
        missing = [name for name in sweep if not (ARTIFACTS_DIR / name).is_file()]
        assert not missing, "release artifacts missing: %s" % missing
        for name in sweep:
            _load(name)

    def test_series_metadata_formats(self):
        """Scope and system-map metadata keep their declared formats."""
        assert _load("production_scope.json")["format"] == "veya-production-scope-v1"
        assert _load("system_map.json")["format"] == "veya-system-map-v1"
        assert _load("phase_audit.json")["release_manifest"] == "artifacts/release/manifest.json"
