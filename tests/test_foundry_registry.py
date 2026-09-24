from pathlib import Path

from veya.foundry import VeyaSourceRegistry


MANIFEST = Path(__file__).resolve().parents[1] / "research" / "provenance" / "sources.json"


def test_registry_has_seventeen_sources():
    registry = VeyaSourceRegistry.from_json(MANIFEST)
    assert len(registry.all()) == 17
    assert len(registry.integrable()) == 15
    assert len(registry.reference_only()) == 2


def test_source_ids_are_opaque_and_stable():
    registry = VeyaSourceRegistry.from_json(MANIFEST)
    assert [r.source_id for r in registry.all()] == [f"source_{i:03d}" for i in range(1, 18)]
