"""Reproducible release metadata assembler (G22).

Collects all inputs that determine a Veya release artifact: git commit,
dependency lock, model/policy/calibration hashes, scorer IDs, SBOM, CI
context, and evaluation references. The working tree is recorded as it
actually is, including any drift between the checked-in release manifest
runtime pins and the files on disk, so another environment can trace a
deployed artifact back to its exact inputs.

Usage:
    python -m research.veyabench.release_metadata [--output PATH]
"""
import datetime
import hashlib
import json
import os
import pathlib
import platform
import subprocess

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
ARTIFACTS = REPO_ROOT / "artifacts" / "release"

METADATA_FORMAT = "veya-release-metadata-v1"

# JSON records in artifacts/release/ that already have a dedicated section
# (or are this record itself) and must not be duplicated in
# evaluation_references.
CORE_ARTIFACTS = frozenset({
    "manifest.json",
    "selective_policy.json",
    "sbom.json",
    "production_scope.json",
    "release_metadata.json",
})


def _sha256_file(path: pathlib.Path) -> str:
    """SHA-256 hash of a file (1MB chunks)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: pathlib.Path) -> dict:
    """Parse a JSON artifact, or {} when the file is absent."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _utc_timestamp() -> str:
    """Current UTC time as seconds-precision ISO 8601 (matches build_sbom.py)."""
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_commit() -> str:
    """Current git HEAD commit SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=REPO_ROOT,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    """Whether the working tree has uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, cwd=REPO_ROOT,
            timeout=5,
        )
        return bool(result.stdout.strip())
    except Exception:
        return True  # assume dirty if it cannot be checked


def _runtime_drift(manifest: dict) -> dict:
    """Manifest runtime pins versus the actual working tree, recorded verbatim."""
    pinned = manifest.get("runtime_files") or {}
    mismatched, missing = [], []
    current_hashes = {}
    for rel_path in sorted(pinned):
        path = REPO_ROOT / rel_path
        if not path.is_file():
            missing.append(rel_path)
            continue
        digest = _sha256_file(path)
        current_hashes[rel_path] = digest
        if digest != pinned[rel_path]:
            mismatched.append(rel_path)
    return {
        "runtime_files_checked": len(pinned),
        "runtime_files_mismatched": mismatched,
        "runtime_files_missing": missing,
        "runtime_file_hashes": current_hashes,
        "note": (
            "manifest.json pins SHA-256 for every served runtime file. Entries in "
            "runtime_files_mismatched are working-tree edits made after the manifest "
            "was minted (post-audit drift); runtime_file_hashes records their accurate "
            "current hashes so the next manifest re-mint can absorb them."
        ),
    }


def _evaluation_references() -> dict:
    """Every JSON artifact in artifacts/release/ not pinned by a dedicated section."""
    references = {}
    for path in sorted(ARTIFACTS.glob("*.json")):
        if path.name in CORE_ARTIFACTS:
            continue
        references["artifacts/release/%s" % path.name] = _sha256_file(path)
    return references


def _ci_context() -> dict:
    """GitHub Actions context when in CI; explicit 'local' fallbacks otherwise."""
    run_id = os.environ.get("GITHUB_RUN_ID", "local")
    if run_id == "local":
        pipeline_url = "local"
    else:
        pipeline_url = "%s/%s/actions/runs/%s" % (
            os.environ.get("GITHUB_SERVER_URL", ""),
            os.environ.get("GITHUB_REPOSITORY", ""),
            run_id,
        )
    return {
        "pipeline_run_id": run_id,
        "pipeline_url": pipeline_url,
        "triggered_by": os.environ.get("GITHUB_ACTOR", "local"),
        "ref": os.environ.get("GITHUB_REF", "local"),
    }


def assemble_release_metadata() -> dict:
    """Build the complete release metadata record."""
    manifest_path = ARTIFACTS / "manifest.json"
    policy_path = ARTIFACTS / "selective_policy.json"
    sbom_path = ARTIFACTS / "sbom.json"
    scope_path = ARTIFACTS / "production_scope.json"
    lock_path = REPO_ROOT / "requirements.lock"

    manifest = _load_json(manifest_path)
    policy = _load_json(policy_path)
    sbom = _load_json(sbom_path)
    scope = _load_json(scope_path)
    calibration = manifest.get("calibration") or {}
    timestamp = _utc_timestamp()

    return {
        "format": METADATA_FORMAT,
        "generated": timestamp,

        # Source control
        "source": {
            "git_commit": _git_commit(),
            "git_dirty": _git_dirty(),
            "repository": "veya",
        },

        # Dependencies
        "dependencies": {
            "lock_file": "requirements.lock",
            "lock_hash": _sha256_file(lock_path) if lock_path.exists() else "missing",
            "sbom_file": "artifacts/release/sbom.json",
            "sbom_hash": _sha256_file(sbom_path) if sbom_path.exists() else "missing",
            "python_version": sbom.get("python_version", "unknown"),
        },

        # Model artifacts — exact manifest/policy field names of the pinned release
        "model_artifacts": {
            "manifest_file": "artifacts/release/manifest.json",
            "manifest_hash": _sha256_file(manifest_path) if manifest_path.exists() else "missing",
            "release": manifest.get("release", "unknown"),
            "checkpoint_path": manifest.get("base_checkpoint", "unknown"),
            "checkpoint_hash": manifest.get("base_checkpoint_hash", "unknown"),
            "shortlist_hash": policy.get("shortlist_sha256", "unknown"),
            "policy_hash": _sha256_file(policy_path) if policy_path.exists() else "missing",
        },

        # Working-tree drift against the manifest runtime pins
        "drift": _runtime_drift(manifest),

        # Calibration — mirrors manifest.calibration field-for-field
        "calibration": {
            "packed_choice": calibration.get("packed_choice", "unknown"),
            "packed_choice_hash": calibration.get("packed_choice_hash", "unknown"),
            "packed_scorer_id": calibration.get("packed_scorer_id", "unknown"),
            "hierarchical_choice": calibration.get("hierarchical_choice", "unknown"),
            "hierarchical_choice_hash": calibration.get("hierarchical_choice_hash", "unknown"),
            "hierarchical_route": calibration.get("hierarchical_route", "unknown"),
            "hierarchical_scorer_id": calibration.get("hierarchical_scorer_id", "unknown"),
            "hierarchical_temperature": calibration.get("hierarchical_temperature", "unknown"),
            "noul": calibration.get("noul", "unknown"),
            "score": calibration.get("score", "unknown"),
        },

        # Scorer/config IDs
        "config": {
            "scorer_id": policy.get("scorer_id", "unknown"),
            "score_method": policy.get("score", "unknown"),
            "route": policy.get("route", "unknown"),
            "operating_epsilon": policy.get("operating_epsilon", "unknown"),
            "architecture_version": manifest.get("architecture_version", "unknown"),
        },

        # CI/CD context (from environment)
        "ci": _ci_context(),

        # Build
        "build": {
            "timestamp": timestamp,
            "docker_image_digest": os.environ.get("DOCKER_IMAGE_DIGEST", "not_built"),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },

        # Evaluation references — every other JSON release artifact, hashed
        "evaluation_references": _evaluation_references(),

        # Production scope
        "production_scope": {
            "scope_file": "artifacts/release/production_scope.json",
            "version": scope.get("version", "unknown"),
            "release": scope.get("release", "unknown"),
        },
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ARTIFACTS / "release_metadata.json"))
    args = parser.parse_args()

    metadata = assemble_release_metadata()
    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Release metadata written to {out}")
    drift = metadata["drift"]
    print(
        "  commit %s (dirty=%s) | evaluation references: %d | drift: %d mismatched, "
        "%d missing of %d runtime files"
        % (
            metadata["source"]["git_commit"][:12],
            metadata["source"]["git_dirty"],
            len(metadata["evaluation_references"]),
            len(drift["runtime_files_mismatched"]),
            len(drift["runtime_files_missing"]),
            drift["runtime_files_checked"],
        )
    )
