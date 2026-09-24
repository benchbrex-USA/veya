"""Opaque source registry for the Veya Foundry."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class VeyaSourceRecord:
    source_id: str
    repository: str
    branch: str
    code_license: str
    integration_policy: str
    capabilities: tuple[str, ...]

    @property
    def integration_allowed(self) -> bool:
        return self.integration_policy == "integrate"


class VeyaSourceRegistry:
    def __init__(self, records: Iterable[VeyaSourceRecord]):
        records = tuple(records)
        ids = [record.source_id for record in records]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate source_id in Veya source registry")
        self._records = {record.source_id: record for record in records}

    @classmethod
    def from_json(cls, path: str | Path) -> "VeyaSourceRegistry":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            VeyaSourceRecord(
                source_id=str(item["source_id"]),
                repository=str(item["repository"]),
                branch=str(item.get("branch") or "main"),
                code_license=str(item.get("code_license") or "unknown"),
                integration_policy=str(item.get("integration_policy") or "reference_only"),
                capabilities=tuple(str(v) for v in item.get("capabilities") or ()),
            )
            for item in payload["sources"]
        )

    def get(self, source_id: str) -> VeyaSourceRecord:
        return self._records[source_id]

    def all(self) -> tuple[VeyaSourceRecord, ...]:
        return tuple(self._records[k] for k in sorted(self._records))

    def integrable(self) -> tuple[VeyaSourceRecord, ...]:
        return tuple(record for record in self.all() if record.integration_allowed)

    def reference_only(self) -> tuple[VeyaSourceRecord, ...]:
        return tuple(record for record in self.all() if not record.integration_allowed)
