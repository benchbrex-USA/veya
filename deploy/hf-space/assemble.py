"""Stage the Hugging Face Space upload tree and re-verify the audited release.

Run this before every upload::

    python3 deploy/hf-space/assemble.py            # build build/hf-space/
    python3 deploy/hf-space/assemble.py --check     # verify only, no writes

The staging directory mirrors the repository layout, because
``web/model_server.py`` derives the release root as the parent of ``web/`` and
``veya/release_bundle.py:load_release_manifest`` resolves every manifest path
relative to that root. Anything that moves, moves the release root with it.

Three things are checked here that a plain ``cp -r`` would not catch:

1. **Every manifest-pinned digest is re-computed against the staged bytes**,
   using the same functions the model service uses at startup
   (``veya.shortlist_artifact.file_sha256`` and
   ``veya.release_bundle.resolve_manifest_path``). A file that is correct in the
   repository but corrupted or stale in staging fails here rather than four
   minutes into a cold start on the free tier.
2. **The checkpoint is asserted absent.** ``models/production/model.safetensors``
   lives in a private model repository and is fetched at boot by
   ``space/fetch_checkpoint.py``. Staging it would republish private research
   weights as public LFS blobs.
3. **Nothing on the deny list is staged.** The source repository is private
   (``gh repo view`` reports ``visibility: PRIVATE``) and the Space is not, so
   every staged file becomes publicly browsable. Legal review, license
   inventory, SBOM, readiness matrices, activation procedure, and dataset
   lineage are audit inputs for humans reviewing the release — they are not
   serving dependencies, and publishing them is a decision, not a side effect.

Manifest references that are deliberately *not* staged are reported rather than
silently dropped, so the deployed tree's dangling references are explicit.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import shutil
import sys
from pathlib import Path

SPACE_DIR = Path(__file__).resolve().parent
REPO = SPACE_DIR.parents[1]
STAGE = REPO / "build" / "hf-space"

# Hugging Face stores files above this in Git LFS. Nothing here should be
# anywhere near it; a hit means something large was staged by accident.
MAX_PLAIN_FILE_BYTES = 10 * 1024 * 1024

# ---------------------------------------------------------------------------
# What the container actually needs
# ---------------------------------------------------------------------------

# Runtime-required, verified by load_release_manifest before the model serves.
PINNED_REQUIRED = True

# Serving code. web/static is read at request time (site-config.json and
# index.html at import, everything else per request).
SERVING_TREES = ("veya", "web")

# Small model-directory companions of the checkpoint: config, tokenizer,
# router, shortlist projection, and the two calibrations the manifest pins.
# model.safetensors itself is fetched at boot; multilingual/ and
# typed-decisions/ are separate 600-800 MB checkpoints that this release does
# not serve; .cache/ holds the download metadata that space/fetch_checkpoint.py
# documents but does not need.
MODEL_FILES = (
    "models/production/rl_agent_config.json",
    "models/production/calibration.json",
    "models/production/calibration_shortlist_k8.json",
    "models/production/choice_router.json",
    "models/production/shortlist_k8.pt",
    "models/production/shortlist_k8.pt.json",
    "models/production/encoder/config.json",
    "models/production/tokenizer/tokenizer.json",
    "models/production/tokenizer/tokenizer_config.json",
)

# artifacts/release files the running service reads or the manifest pins.
#   manifest.json                  - the release gate itself
#   selective_policy.json          - hash-pinned, loaded at startup
#   route_cutoff_audit.json        - manifest routing.cutoff_evidence
#   blind_clinc150_val_result.json - read per request by /api/benchmark
#                                    (web/server.py); without it the live
#                                    benchmarks page 503s
RELEASE_FILES = (
    "artifacts/release/manifest.json",
    "artifacts/release/selective_policy.json",
    "artifacts/release/route_cutoff_audit.json",
    "artifacts/release/blind_clinc150_val_result.json",
)

BUILD_INPUTS = ("requirements.lock", "pyproject.toml", "setup.py")

# Never staged, even if a glob or a future edit would pull them in.
DENY_GLOBS = (
    "*.safetensors",
    "*.pt.json.bak",
    ".env",
    ".env.*",
    "*.pypirc",
    ".pypirc",
    "*.token",
    ".hf_token",
    "hf_token*",
    "secrets.json",
    "credentials.json",
    "session-*.md",
    "*-session.md",
    ".claude-session*",
    "*-local-command-*.txt",
    "__pycache__/*",
    "*.pyc",
    ".DS_Store",
    # Audit, legal, and supply-chain material. Not serving dependencies, and
    # the source repository is private.
    "artifacts/release/legal_review_packet.json",
    "artifacts/release/license_inventory.json",
    "artifacts/release/sbom.json",
    "artifacts/release/production_readiness_matrix.json",
    "artifacts/release/production_scope.json",
    "artifacts/release/gap_audit.json",
    "artifacts/release/activation_procedure.json",
    "artifacts/release/shadow_schema.json",
    "artifacts/release/system_map.json",
    "artifacts/release/release_metadata.json",
    "artifacts/release/prospective_blind_config.template.json",
    "artifacts/release/runtime_audit.json",
    "artifacts/release/phase_audit.json",
    "artifacts/release/manifest-phase5.json",
    "artifacts/release/manifest-phase9.json",
    "artifacts/release/blind_clinc150_test_failed_run.json",
    "docs/*",
    "research/*",
    "tests/*",
    "notebooks/*",
    "k8s/*",
    ".github/*",
    "models/production/multilingual/*",
    "models/production/typed-decisions/*",
    "models/production/.cache/*",
    "models/production/calibration_multilingual.json",
    "models/production/calibration_noul.json",
    "models/production/calibration_score.json",
    "models/production/calibration_hypothesis_hce2.json",
    # Internal audit write-ups (web/PRELAUNCH_AUDIT.md, VERIFICATION_REPORT.md,
    # EXECUTION_GRAPH.md) record verdicts about a past working tree, including
    # "blocked for public production release". Served next to a live service
    # they would read as a current verdict on it. The Space's own root README
    # describes the deployed release instead.
    "web/*.md",
)


class AssembleError(RuntimeError):
    pass


def denied(rel: str) -> str | None:
    """Return the deny pattern that matches ``rel``, or None."""
    for pattern in DENY_GLOBS:
        if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(Path(rel).name, pattern):
            return pattern
    return None


def staged_paths() -> list[Path]:
    return sorted(p for p in STAGE.rglob("*") if p.is_file())


def rel_in_stage(path: Path) -> str:
    return path.relative_to(STAGE).as_posix()


def copy_one(src: Path, rel: str) -> None:
    if not src.is_file():
        raise AssembleError(f"missing source file: {rel}")
    match = denied(rel)
    if match is not None:
        raise AssembleError(f"refusing to stage {rel} (matches deny pattern {match!r})")
    dest = STAGE / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    dest.chmod(0o644)


def copy_tree(rel: str) -> int:
    """Copy a serving tree, skipping caches and anything on the deny list."""
    root = REPO / rel
    if not root.is_dir():
        raise AssembleError(f"missing source tree: {rel}")
    count = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        child = path.relative_to(REPO).as_posix()
        if "__pycache__" in path.parts:
            continue
        if denied(child) is not None:
            continue
        copy_one(path, child)
        count += 1
    if count == 0:
        raise AssembleError(f"staged nothing from {rel}")
    return count


def build() -> dict[str, int]:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)

    counts: dict[str, int] = {}

    # Space-only scaffolding: these two files are not the repository's own
    # Dockerfile and README, they are the Space's. They land at the staging root
    # because that is the Space repository root.
    for name in ("Dockerfile", "README.md"):
        copy_one(SPACE_DIR / name, name)
        counts.setdefault("space_scaffolding", 0)
        counts["space_scaffolding"] += 1
    for name in ("boot.py", "fetch_checkpoint.py"):
        copy_one(SPACE_DIR / "space" / name, f"space/{name}")
        counts["space_scaffolding"] += 1

    for rel in BUILD_INPUTS:
        copy_one(REPO / rel, rel)
        counts.setdefault("build_inputs", 0)
        counts["build_inputs"] += 1

    for rel in SERVING_TREES:
        counts[rel] = copy_tree(rel)

    for group, names in (("release_artifacts", RELEASE_FILES), ("model_files", MODEL_FILES)):
        counts[group] = 0
        for rel in names:
            copy_one(REPO / rel, rel)
            counts[group] += 1

    return counts


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def manifest_references(manifest: dict) -> list[tuple[str, str]]:
    """Every filesystem path the manifest names, as (json_pointer, relative_path).

    A string is treated as a path only if it looks like one: it has a file
    extension, or it names a directory that exists in the repository. That
    filters out values such as ``git_branch`` ("codex/phase-1-9-forensic-audit")
    and ``served_scorer_id``, which contain a slash but are identifiers.
    """
    found: list[tuple[str, str]] = []

    def is_path(value: str) -> bool:
        if not value or value.startswith("http") or "/" not in value:
            return False
        if Path(value).suffix:
            return True
        return (REPO / value).is_dir()

    def walk(node, pointer: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{pointer}.{key}" if pointer else key)
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{pointer}[{index}]")
        elif isinstance(node, str) and is_path(node):
            found.append((pointer, node))

    walk(manifest, "")
    return found


def verify(checkpoint_expected_absent: bool = True) -> list[str]:
    """Re-verify the staged tree. Returns manifest references left dangling."""
    # Imported here, not at module scope: these are the production helpers, and
    # pulling them in means importing torch. Failing loudly is correct — an
    # assembler that re-implements the digest would verify against a different
    # definition of "correct" than the model service does.
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from veya.release_bundle import resolve_manifest_path
    from veya.shortlist_artifact import file_sha256

    if not STAGE.is_dir():
        raise AssembleError(f"nothing staged at {STAGE}; run without --check first")

    staged = staged_paths()
    if not staged:
        raise AssembleError(f"staging directory is empty: {STAGE}")

    problems: list[str] = []
    oversized: list[str] = []
    for path in staged:
        rel = rel_in_stage(path)
        match = denied(rel)
        if match is not None:
            problems.append(f"staged file matches deny pattern {match!r}: {rel}")
        size = path.stat().st_size
        if size > MAX_PLAIN_FILE_BYTES:
            oversized.append(f"{rel} ({size / 1e6:.1f} MB)")
    if oversized:
        problems.append("files above the 10 MB plain-git limit would need LFS: " + ", ".join(oversized))

    # The checkpoint must not be here: it is fetched at boot from the private
    # model repository. The generic sweep below also catches any other weights
    # file, but this one names the specific failure mode.
    checkpoint = STAGE / "models/production/model.safetensors"
    if checkpoint_expected_absent and checkpoint.exists():
        problems.append(
            "models/production/model.safetensors is staged; it must be fetched "
            "at boot from the private model repository instead"
        )
    any_weights = [rel_in_stage(p) for p in staged if p.suffix == ".safetensors"]
    if any_weights:
        problems.append(f"safetensors files staged: {', '.join(any_weights)}")

    # Re-verify every pinned digest against the staged bytes, resolving through
    # the production helper so a path that escapes the release root is caught the
    # same way it would be at startup.
    manifest_path = STAGE / "artifacts/release/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stage_root = str(STAGE)

    pinned: list[tuple[str, str]] = []  # (relative_path, expected_hash)
    for rel, digest in (manifest.get("runtime_files") or {}).items():
        pinned.append((rel, digest))
    for rel, digest in (manifest.get("tokenizer_files") or {}).items():
        pinned.append((rel, digest))
    pinned.append((manifest["model_config"], manifest["model_config_hash"]))
    routing = manifest.get("routing") or {}
    pinned.append((routing["choice_router"], routing["choice_router_hash"]))
    choice = manifest.get("choice") or {}
    pinned.append((choice["shortlist_artifact"], choice["shortlist_hash"]))
    calibration = manifest.get("calibration") or {}
    pinned.append((calibration["packed_choice"], calibration["packed_choice_hash"]))
    pinned.append((calibration["hierarchical_choice"], calibration["hierarchical_choice_hash"]))
    pinned.append((manifest["selective_policy"], manifest["selective_policy_hash"]))

    verified = 0
    for rel, expected in pinned:
        try:
            resolved = resolve_manifest_path(stage_root, rel)
        except ValueError as exc:
            problems.append(f"manifest path {rel} does not resolve inside staging: {exc}")
            continue
        if not Path(resolved).is_file():
            problems.append(f"pinned file missing from staging: {rel}")
            continue
        actual = file_sha256(resolved)
        if actual != expected:
            problems.append(f"pinned digest mismatch for {rel}: staged {actual[:16]}… expected {expected[:16]}…")
            continue
        verified += 1

    # The checkpoint hash cannot be verified from staging — the file is fetched
    # at boot. Confirm the manifest still carries a plausible 64-hex digest so
    # space/fetch_checkpoint.py has something to assert against.
    checkpoint_hash = manifest.get("base_checkpoint_hash")
    if not isinstance(checkpoint_hash, str) or len(checkpoint_hash) != 64:
        problems.append(f"manifest base_checkpoint_hash is not a 64-hex digest: {checkpoint_hash!r}")
    else:
        verified += 1

    # Report, do not fail: these are offline-reproducibility and audit inputs.
    def staged_ok(rel: str) -> bool:
        target = STAGE / rel
        return target.is_file() or (target.is_dir() and any(target.rglob("*")))

    dangling = [
        f"{pointer} -> {rel}"
        for pointer, rel in manifest_references(manifest)
        if rel != manifest["base_checkpoint"] and not staged_ok(rel)
    ]

    if problems:
        raise AssembleError("staged tree failed verification:\n  - " + "\n  - ".join(problems))

    total = sum(p.stat().st_size for p in staged)
    print(f"staged {len(staged)} files, {total / 1e6:.2f} MB -> {STAGE}")
    print(f"verified {verified} manifest digests against the staged bytes")
    print(f"checkpoint: absent as required (boot fetch asserts {str(checkpoint_hash)[:16]}…)")
    if dangling:
        print(f"\nmanifest references deliberately not staged ({len(dangling)}):")
        for entry in dangling:
            print(f"  - {entry}")
        print(
            "  These are dataset-lineage, blind-evaluation, and phase-9 research\n"
            "  inputs. load_release_manifest does not verify them, and nothing in\n"
            "  the serving path reads them."
        )
    return dangling


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="verify the existing staging tree without rebuilding it")
    args = parser.parse_args(argv)

    try:
        if not args.check:
            counts = build()
            print("staged: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        verify()
    except AssembleError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"\nnext: upload {STAGE} as the Space repository root.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
