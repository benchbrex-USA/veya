"""Versioned product contracts.

/v1/decisions is the audited Veya typed contract.
/v1/responses is specialist work and does not enter that contract.
"""

from __future__ import annotations

from typing import Any

from web.contracts import RequestError, validate

from product.policy import PolicyError, production_route

SERVED = ("choice", "noul", "score")


class ProductError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


def parse_decision(data: Any) -> dict[str, Any]:
    if isinstance(data, dict) and data.get("primitive") == "systemone":
        raise ProductError("unsupported_primitive", "SystemOne is not served by this release.")
    if not isinstance(data, dict) or data.get("primitive") not in SERVED:
        raise ProductError("invalid_input", "Select choice, noul, or categorical score.")
    try:
        body = validate(data)
    except RequestError as exc:
        raise ProductError("invalid_input", str(exc)) from exc
    return {"endpoint": "/v1/decisions", "primitive": data["primitive"], "body": body}


def parse_response(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ProductError("invalid_input", "Request body must be an object.")
    if "model" in data or "base_url" in data:
        raise ProductError("invalid_input", "Caller cannot select a model or URL.")
    if data.get("task") == "bounded_decision":
        raise ProductError("wrong_endpoint", "Bounded decisions use /v1/decisions.")
    try:
        decision = production_route(data)
    except PolicyError as exc:
        raise ProductError("invalid_input", str(exc)) from exc
    if decision["selected"] != "veya-core":
        return {
            "endpoint": "/v1/responses",
            "code": "specialist_not_enabled",
            "detail": "No production specialist is enabled for this task.",
            "route": decision,
        }
    return {"endpoint": "/v1/responses", "route": decision}
