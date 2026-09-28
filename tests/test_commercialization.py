import json
from pathlib import Path

from veya.shortlist_artifact import file_sha256


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


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
    policy = read_json("artifacts/commercialization/query_policy_freeze.json")
    decision = read_json("artifacts/decisions/commercialization.json")
    blind = read_json("artifacts/commercialization/blind_hwu64_result.json")

    assert policy["status"] == "FROZEN_FOR_NEW_BLIND_NOT_PRODUCTION"
    assert policy["policy"]["query_fusion"] == "max"
    assert policy["production_default_changed"] is False
    assert policy["artifact_sha256"] == file_sha256(
        str(ROOT / "artifacts/experiments/phase10/domain_general_k8.pt")
    )
    assert policy["development_report_sha256"] == file_sha256(
        str(ROOT / "artifacts/commercialization/query_robustness.json")
    )

    assert blind["status"] == "BLIND_COMPLETE_CONSUMED"
    assert blind["frozen"]["policy_sha256"] == file_sha256(
        str(ROOT / "artifacts/commercialization/query_policy_freeze.json")
    )
    assert blind["frozen"]["production_manifest_sha256"] == file_sha256(
        str(ROOT / "artifacts/release/manifest.json")
    )
    assert blind["dataset"]["license_sha256"] == file_sha256(
        str(ROOT / "research/data/hwu64/LICENSE")
    )

    assert decision["commercial_verdict"] == "RESEARCH"
    assert decision["production_promotion"] is False
    assert decision["production_manifest_sha256"] == file_sha256(
        str(ROOT / "artifacts/release/manifest.json")
    )
