"""Production routing policy.

External specialists stay disabled until a promotion record enables one by id.
Veya Core remains the only bounded-decision model.
"""

from __future__ import annotations

from typing import Any

from experts.registry import VEYA, public_registry
from experts.router import RouteError, route

# Empty on purpose. Listing a model in the registry is not production enablement.
PRODUCTION_ENABLED: frozenset[str] = frozenset()


class PolicyError(ValueError):
    pass


def production_route(request: dict[str, Any], enabled: set[str] | None = None) -> dict[str, Any]:
    try:
        decision = route(request, enabled if enabled is not None else set(PRODUCTION_ENABLED))
    except RouteError as exc:
        raise PolicyError(str(exc)) from exc
    decision["production_enabled"] = decision["selected"] == VEYA
    decision["benchbrex_tuned"] = False
    if decision["invokes_external"]:
        decision["callable"] = False
        decision["selected"] = "defer"
        decision["reason"] = "specialist is not production-enabled"
    return decision


def model_catalog() -> list[dict[str, Any]]:
    """Public capability list. Release claims are not upgraded here."""
    rows = []
    for item in public_registry():
        model_id = item["model_id"]
        profile = item["profile"]
        rows.append(
            {
                "model_id": model_id,
                "role": item["role"],
                "tasks": list(item["tasks"]),
                "privacy_class": item["privacy_class"],
                "production_enabled": model_id == VEYA,
                "benchbrex_tuned": bool(profile.get("benchbrex_tuned")),
                "profile_name": profile.get("profile_name"),
                "profile_source": profile.get("source"),
            }
        )
    return rows
