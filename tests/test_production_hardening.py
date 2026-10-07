"""Hardening for the edge-managed container: tenant-keyed limiter, bounded cache, CPU threads."""

import ast
import os
import re
from pathlib import Path

import numpy as np

from web import server as gateway
from veya import release_bundle

ROOT = Path(__file__).resolve().parents[1]


def test_without_edge_management_the_limiter_keys_on_the_socket_ip(monkeypatch):
    monkeypatch.delenv("VEYA_EDGE_MANAGED", raising=False)
    assert gateway.limit_key("10.0.0.1", "key_1") == ("10.0.0.1", gateway.CAPACITY)


def test_edge_managed_limiter_keys_on_the_tenant_not_the_shared_ip(monkeypatch):
    monkeypatch.setenv("VEYA_EDGE_MANAGED", "1")
    monkeypatch.delenv("VEYA_EDGE_TENANT_CAPACITY", raising=False)
    key, capacity = gateway.limit_key("10.0.0.1", "key_1")
    assert key == "tenant:key_1"
    assert capacity >= 20, "the in-container cap must not be tighter than the edge per-key limit"
    # a missing or malformed tenant falls back to the IP bucket
    assert gateway.limit_key("10.0.0.1", None)[0] == "10.0.0.1"
    assert gateway.limit_key("10.0.0.1", "bad tenant/../")[0] == "10.0.0.1"
    assert gateway.limit_key("10.0.0.1", "x" * 97)[0] == "10.0.0.1"


def test_two_tenants_behind_one_ip_do_not_share_a_bucket(monkeypatch):
    monkeypatch.setenv("VEYA_EDGE_MANAGED", "1")
    monkeypatch.setenv("VEYA_EDGE_TENANT_CAPACITY", "12")
    gateway.rate_hits.clear()
    try:
        first, capacity = gateway.limit_key("10.0.0.1", "tenant_a")
        for _ in range(capacity):
            assert gateway._charge(first, 1, capacity) == 0
        assert gateway._charge(first, 1, capacity) > 0
        second, capacity = gateway.limit_key("10.0.0.1", "tenant_b")
        assert gateway._charge(second, 1, capacity) == 0
    finally:
        gateway.rate_hits.clear()


def test_tenant_capacity_is_bounded(monkeypatch):
    monkeypatch.setenv("VEYA_EDGE_TENANT_CAPACITY", "1")
    assert gateway._edge_tenant_capacity() == gateway.CAPACITY
    monkeypatch.setenv("VEYA_EDGE_TENANT_CAPACITY", "100000")
    assert gateway._edge_tenant_capacity() == 600
    monkeypatch.setenv("VEYA_EDGE_TENANT_CAPACITY", "nope")
    assert gateway._edge_tenant_capacity() == 30


def test_container_class_turns_on_edge_management_and_edge_strips_client_tenant():
    container = (ROOT / "worker" / "veya-container.mjs").read_text(encoding="utf-8")
    assert 'VEYA_EDGE_MANAGED: "1"' in container
    policy = (ROOT / "worker" / "edge_policy.mjs").read_text(encoding="utf-8")
    hop = policy.split("const HOP = [", 1)[1].split("];", 1)[0]
    assert '"x-veya-tenant"' in hop


class _Bundle(release_bundle.ReleaseBundle):
    def __init__(self, cap):  # noqa: D107 - skip model loading, keep the cache fields
        self.router = {"embed_max_length": 8}
        self._cache = release_bundle.OrderedDict()
        self._cache_lock = release_bundle.threading.Lock()
        self._cache_max = cap
        self.calls = []

        def embed(texts):
            self.calls.append(list(texts))
            return [np.full(4, float(len(text)), dtype=np.float32) for text in texts]

        self._embed = embed


def test_embedding_cache_is_a_bounded_lru():
    bundle = _Bundle(cap=3)
    out = bundle.embed(["a", "bb", "ccc"])
    assert out.shape == (3, 4)
    bundle.embed(["a"])  # refresh "a"
    bundle.embed(["dddd"])  # evicts the least recently used: "bb"
    assert list(bundle._cache) == ["ccc", "a", "dddd"]
    assert len(bundle._cache) == 3
    bundle.embed(["bb"])
    assert bundle.calls[-1] == ["bb"]
    assert len(bundle._cache) == 3


def test_embedding_cache_returns_identical_vectors_and_dedupes():
    bundle = _Bundle(cap=10)
    first = bundle.embed(["x", "yy", "x"])
    assert bundle.calls == [["x", "yy"]]
    again = bundle.embed(["x", "yy", "x"])
    assert bundle.calls == [["x", "yy"]]
    np.testing.assert_array_equal(first, again)


def test_embed_cache_max_env(monkeypatch):
    assert release_bundle.embed_cache_max({}) == release_bundle.DEFAULT_EMBED_CACHE_MAX
    assert release_bundle.embed_cache_max({"VEYA_EMBED_CACHE_MAX": "123"}) == 123
    assert release_bundle.embed_cache_max({"VEYA_EMBED_CACHE_MAX": "0"}) == 1
    assert release_bundle.embed_cache_max({"VEYA_EMBED_CACHE_MAX": "x"}) == release_bundle.DEFAULT_EMBED_CACHE_MAX


def _cpu_threads():
    """Extract cpu_threads from web/model_server.py without loading the checkpoint."""
    source = (ROOT / "web" / "model_server.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == "cpu_threads")
    namespace = {"os": os}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "model_server.py", "exec"), namespace)
    return namespace["cpu_threads"], source


def test_model_server_pins_torch_threads_before_loading():
    cpu_threads, source = _cpu_threads()
    assert cpu_threads({"VEYA_TORCH_THREADS": "3"}) == 3
    assert cpu_threads({"VEYA_TORCH_THREADS": "500"}) == 64
    assert cpu_threads({"VEYA_TORCH_THREADS": "0"}) >= 1
    assert cpu_threads({}) >= 1
    assert re.search(r"torch\.set_num_threads\(cpu_threads\(\)\)", source)
    assert source.index("torch.set_num_threads") < source.index("load_release_manifest(")


def test_gateway_upstream_paths_are_fixed_constants():
    from web import server

    assert server.UPSTREAM_INFER_PATHS == {
        "choice": "/v1/choice",
        "noul": "/v1/noul",
        "score": "/v1/score",
    }
    assert "systemone" not in server.UPSTREAM_INFER_PATHS


def test_audit_text_never_emits_line_breaks():
    from web import server

    cleaned = server._audit_text("/v1/x\r\nINJECTED: yes\u2028more", 300)
    assert "\n" not in cleaned and "\r" not in cleaned
    assert cleaned.startswith("/v1/x__INJECTED")


def test_codeql_gate_allowlist_is_narrow_and_justified(tmp_path):
    import importlib.util
    import json as _json

    spec = importlib.util.spec_from_file_location(
        "codeql_gate", ROOT / ".github" / "scripts" / "codeql_gate.py"
    )
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    allowed = gate.load_allowlist()
    assert allowed == {
        ("py/weak-sensitive-data-hashing", "product/keys.py"),
        ("py/clear-text-logging-sensitive-data", "tests/test_local_e2e.py"),
    }

    def result(rule, uri, level=None):
        out = {
            "ruleId": rule,
            "message": {"text": "m"},
            "locations": [{"physicalLocation": {"artifactLocation": {"uri": uri}, "region": {"startLine": 1}}}],
        }
        if level:
            out["level"] = level
        return out

    sarif = {"runs": [{"tool": {"driver": {"rules": []}}, "results": [
        result("py/weak-sensitive-data-hashing", "product/keys.py"),
        result("py/weak-sensitive-data-hashing", "web/server.py"),
        result("py/log-injection", "web/server.py", "error"),
        result("py/unused-import", "veya/x.py", "note"),
    ]}]}
    found = gate.findings(sarif, allowed)
    assert len(found) == 2
    (tmp_path / "a.sarif").write_text(_json.dumps(sarif))
    assert gate.main(["codeql_gate.py", str(tmp_path)]) == 1
