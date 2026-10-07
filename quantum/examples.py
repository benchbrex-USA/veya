"""Bounded, project-scoped example retrieval and an optional observation cache.

Example bytes are not copied into telemetry. The cache key includes the
project id so two projects cannot share a slot.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from quantum.config import PROMPT_VERSION, SCHEMA_VERSION


class ExampleIndex:
    def __init__(self, records: list[dict[str, Any]] | None = None):
        self._records = list(records or [])

    def resolve(
        self,
        *,
        project_id: str,
        example_ids: list[str],
        family: str,
        device_type: str | None,
        k: int,
        retrieve: bool,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        limit = max(0, min(8, k))
        found: list[dict[str, Any]] = []
        missing: list[str] = []
        for example_id in example_ids[:limit]:
            match = self._find(project_id, example_id)
            if match is None:
                missing.append(example_id)
            else:
                found.append(_public_example(match))
        if retrieve and len(found) < limit:
            chosen = {item["id"] for item in found}
            ranked = self._search(project_id, family, device_type)
            for record in ranked:
                if record["id"] in chosen:
                    continue
                found.append(_public_example(record))
                if len(found) >= limit:
                    break
        return found[:limit], missing

    def _find(self, project_id: str, example_id: str) -> dict[str, Any] | None:
        for record in self._records:
            if record.get("id") == example_id and record.get("project_id") == project_id:
                return record
        return None

    def _search(self, project_id: str, family: str, device_type: str | None) -> list[dict[str, Any]]:
        scored = []
        for record in self._records:
            if record.get("project_id") != project_id:
                continue
            score = 0
            if record.get("experiment_family") == family:
                score += 2
            if device_type and record.get("device_type") == device_type:
                score += 1
            if score:
                scored.append((score, str(record.get("id")), record))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [record for _, _, record in scored]


def _public_example(record: dict[str, Any]) -> dict[str, Any]:
    """Drop image bytes. Verified labels stay text."""
    return {
        "id": record.get("id"),
        "experiment_family": record.get("experiment_family"),
        "device_type": record.get("device_type"),
        "measurements": record.get("measurements") or {},
        "verified_diagnosis": record.get("verified_diagnosis"),
        "verified_next_action": record.get("verified_next_action"),
        "outcome": record.get("outcome"),
    }


class ObservationCache:
    def __init__(self, max_entries: int = 32):
        self.max_entries = max_entries
        self._data: dict[str, dict[str, Any]] = {}
        self._order: list[str] = []

    def get(self, key: str) -> dict[str, Any] | None:
        value = self._data.get(key)
        if value is None:
            return None
        return json.loads(json.dumps(value))

    def put(self, key: str, observation: dict[str, Any]) -> None:
        if key in self._data:
            self._data[key] = json.loads(json.dumps(observation))
            return
        self._data[key] = json.loads(json.dumps(observation))
        self._order.append(key)
        while len(self._order) > self.max_entries:
            self._data.pop(self._order.pop(0), None)


def cache_key(
    *,
    model: str,
    project_id: str,
    image_hashes: list[str],
    evidence: dict[str, Any],
    example_ids: list[str],
    mode: str,
) -> str:
    payload = {
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "project_id": project_id,
        "image_sha256": image_hashes,
        "evidence_sha256": hashlib.sha256(_canonical(evidence).encode("utf-8")).hexdigest(),
        "example_ids": example_ids,
        "mode": mode,
    }
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_drop)


def _drop(_: Any) -> str:
    return ""
