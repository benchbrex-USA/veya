import json
import shutil
from pathlib import Path

import pytest

from veya.shortlist_artifact import file_sha256


ROOT = Path(__file__).resolve().parents[1]

# The manifest mint that produced the commercialization-track blind HWU64 evaluation.
# artifacts/release/manifest.json was legitimately re-minted (Task 29, the
# veya/selective.py pin) from c7febd23... to cd1e2029..., so this hash is frozen
# history: it is recorded identically by the blind result, the consuming decision
# record (artifacts/decisions/commercialization.json), and the frozen development
# report (artifacts/commercialization/query_robustness.json), and is documented as a
# residual risk in artifacts/release/production_readiness_matrix.json. Historical
# evidence must never be re-bound to the current manifest; current release integrity
# is separately covered by tests/test_artifact_integrity.py and
# tests/test_release_metadata.py.
PRE_REMINT_MANIFEST_SHA256 = (
    "c7febd238afa79144d907cdc4eb4bd229a0bca705c952352302772f8269d9648"
)

# Immutable historical evidence read by validate_commercialization_history().
HISTORICAL_EVIDENCE_FILES = (
    "artifacts/commercialization/query_policy_freeze.json",
    "artifacts/commercialization/query_robustness.json",
    "artifacts/commercialization/blind_hwu64_result.json",
    "artifacts/decisions/commercialization.json",
    "docs/commercialization/blind_hwu64_freeze.json",
    "artifacts/experiments/phase10/domain_general_k8.pt",
    "research/data/hwu64/LICENSE",
)


def read_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def validate_commercialization_history(root: Path) -> str:
    """Validate the frozen commercialization-track evidence rooted at ``root``.

    The blind HWU64 record is consumed historical evidence (status
    BLIND_COMPLETE_CONSUMED): it stays bound to the manifest mint that produced it,
    corroborated by the consuming decision record and by the development report whose
    bytes the policy freeze pins. This function never reads
    artifacts/release/manifest.json; it returns the historical manifest hash.
    """

    def rj(rel: str) -> dict:
        return json.loads((root / rel).read_text())

    policy = rj("artifacts/commercialization/query_policy_freeze.json")
    robustness = rj("artifacts/commercialization/query_robustness.json")
    blind = rj("artifacts/commercialization/blind_hwu64_result.json")
    decision = rj("artifacts/decisions/commercialization.json")

    assert policy["status"] == "FROZEN_FOR_NEW_BLIND_NOT_PRODUCTION"
    assert policy["policy"]["query_fusion"] == "max"
    assert policy["production_default_changed"] is False
    assert policy["artifact_sha256"] == file_sha256(
        str(root / "artifacts/experiments/phase10/domain_general_k8.pt")
    )
    assert policy["development_report_sha256"] == file_sha256(
        str(root / "artifacts/commercialization/query_robustness.json")
    )

    assert blind["status"] == "BLIND_COMPLETE_CONSUMED"
    assert blind["frozen"]["policy_sha256"] == file_sha256(
        str(root / "artifacts/commercialization/query_policy_freeze.json")
    )
    # The result is bound to its pre-registered protocol (the freeze record) and to
    # the exact frozen retriever artifact that the policy freeze pins.
    assert blind["frozen"]["protocol_sha256"] == file_sha256(
        str(root / "docs/commercialization/blind_hwu64_freeze.json")
    )
    assert blind["frozen"]["retriever_artifact_sha256"] == policy["artifact_sha256"]
    assert blind["dataset"]["license_sha256"] == file_sha256(
        str(root / "research/data/hwu64/LICENSE")
    )

    # The manifest pin records the producing mint, not the current manifest. Three
    # mutually pinned frozen records must agree on it, and it must be the documented
    # pre-re-mint value, so no single record (nor a coordinated rewrite) can drift.
    historical_manifest = blind["frozen"]["production_manifest_sha256"]
    assert historical_manifest == PRE_REMINT_MANIFEST_SHA256
    assert robustness["frozen"]["production_manifest_sha256"] == historical_manifest
    assert (
        robustness["frozen"]["production_release"]
        == blind["frozen"]["production_release"]
    )
    assert decision["production_manifest_sha256"] == historical_manifest

    assert decision["commercial_verdict"] == "RESEARCH"
    assert decision["production_promotion"] is False
    assert decision["production_manifest_changed"] is False
    return historical_manifest


def test_commercialization_graph_is_complete_and_acyclic():
    graph = read_json("artifacts/commercialization/graph.json")
    nodes = graph["nodes"]
    expected = {f"COM-G{index}" for index in range(28)}
    by_id = {node["id"]: node for node in nodes}
    assert set(by_id) == expected
    assert len(by_id) == len(nodes)

    allowed_status = {"pending", "ready", "active", "blocked", "failed", "complete", "skipped"}
    allowed_research = {"IDEA", "IMPLEMENTED", "TRAINED", "VALIDATED", "BLIND-VALIDATED"}
    allowed_commercial = {"NOT SHIPPABLE", "INTERNAL", "DEVELOPER PREVIEW", "PAID BETA", "GA"}
    required = {
        "id",
        "objective",
        "dependencies",
        "acceptance",
        "status",
        "evidence",
        "blocker",
        "research_status",
        "commercial_status",
    }
    for node in nodes:
        assert set(node) == required
        assert node["status"] in allowed_status
        assert node["research_status"] in allowed_research
        assert node["commercial_status"] in allowed_commercial
        assert set(node["dependencies"]).issubset(expected)

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> None:
        assert node_id not in visiting, "commercialization graph contains a cycle"
        if node_id in visited:
            return
        visiting.add(node_id)
        for dependency in by_id[node_id]["dependencies"]:
            visit(dependency)
        visiting.remove(node_id)
        visited.add(node_id)

    for node_id in expected:
        visit(node_id)


def test_commercialization_freezes_match_artifacts():
    """Frozen commercialization evidence matches its immutable historical anchors.

    The blind record's manifest pin deliberately does not reference the current
    artifacts/release/manifest.json: it records the mint that produced the evaluation
    (superseded by the Task 29 re-mint from c7febd23... to cd1e2029...). Current
    manifest integrity is covered separately by tests/test_artifact_integrity.py and
    tests/test_release_metadata.py.
    """
    assert validate_commercialization_history(ROOT) == PRE_REMINT_MANIFEST_SHA256


def test_historical_blind_evidence_is_immutable_and_independent_of_current_manifest(
    tmp_path,
):
    """Regression guard (Task 36): historical evidence must survive any legitimate
    manifest re-mint, while tampering with the historical records must still fail.

    The pre-fix test required blind_hwu64_result.json's manifest pin to equal the
    current artifacts/release/manifest.json, so the legitimate Task 29 re-mint broke
    the suite and invited "fixing" the failure by rewriting frozen history.
    """
    # Independence: sandbox the immutable evidence under an arbitrary different
    # manifest (simulating the current or any future re-mint) and confirm historical
    # validation is unaffected, because it never reads the current manifest.
    sandbox = tmp_path / "veya"
    for rel in HISTORICAL_EVIDENCE_FILES:
        target = sandbox / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, target)
    sandbox_manifest = sandbox / "artifacts/release/manifest.json"
    sandbox_manifest.parent.mkdir(parents=True, exist_ok=True)
    sandbox_manifest.write_text('{"format": "veya-release-manifest-v2", "simulated_remint": true}')
    assert validate_commercialization_history(sandbox) == PRE_REMINT_MANIFEST_SHA256

    # Immutability: re-binding any single historical record to the sandbox "current"
    # manifest must fail.
    tamper_pins = (
        (
            "artifacts/commercialization/blind_hwu64_result.json",
            ("frozen", "production_manifest_sha256"),
        ),
        ("artifacts/decisions/commercialization.json", ("production_manifest_sha256",)),
        (
            "artifacts/commercialization/query_robustness.json",
            ("frozen", "production_manifest_sha256"),
        ),
    )
    forged_hash = file_sha256(str(sandbox_manifest))

    def tamper(rel: str, pin: tuple) -> None:
        record = json.loads((sandbox / rel).read_text())
        node = record
        for key in pin[:-1]:
            node = node[key]
        node[pin[-1]] = forged_hash
        (sandbox / rel).write_text(json.dumps(record, indent=2))

    for rel, pin in tamper_pins:
        pristine = (sandbox / rel).read_text()
        tamper(rel, pin)
        with pytest.raises(AssertionError):
            validate_commercialization_history(sandbox)
        (sandbox / rel).write_text(pristine)
    assert validate_commercialization_history(sandbox) == PRE_REMINT_MANIFEST_SHA256

    # A coordinated rewrite of every corroborating record must still fail: the frozen
    # pre-re-mint pin is the terminal anchor.
    for rel, pin in tamper_pins:
        tamper(rel, pin)
    with pytest.raises(AssertionError):
        validate_commercialization_history(sandbox)
