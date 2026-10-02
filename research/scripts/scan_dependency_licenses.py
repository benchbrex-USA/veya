#!/usr/bin/env python3
"""Dependency license scanner for the G15 licensing compliance inventory.

Cross-references the runtime dependencies declared in ``pyproject.toml``
(``[project].dependencies``) against the packages installed in the running
environment and reports version and license metadata for each. Installed
packages outside the declared runtime set (dev/test tooling, optional
extras, the project itself) are reported separately as excluded, so the
captured output is a complete, auditable picture of the environment.

The scan is read-only. Its JSON output feeds the ``runtime_dependencies``
section of ``artifacts/release/license_inventory.json``; model and dataset
assets are inventoried separately from repository evidence.

Usage:
    .venv/bin/python research/scripts/scan_dependency_licenses.py
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from importlib import metadata as md
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("Python 3.11+ is required (tomllib not available)") from exc

ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = ROOT / "pyproject.toml"

# G15 status policy: only these license families count as CLEARED. Anything
# copyleft, unknown/missing, or outside this set is REQUIRES_LEGAL_REVIEW.
PERMISSIVE_MARKERS = (r"\bmit\b", r"\bbsd\b", r"\bapache\b", r"\bisc\b")
COPYLEFT_MARKERS = (r"\bgpl\b", r"\blgpl\b", r"\bagpl\b", r"\baffero\b", r"share-?alike")
REQUIREMENT_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def canonical(name: str) -> str:
    """PEP 503 name normalization."""
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_name(requirement: str) -> str:
    match = REQUIREMENT_NAME.match(requirement)
    if not match:
        raise SystemExit(f"cannot parse requirement: {requirement!r}")
    return match.group(1)


def dist_license(dist: md.Distribution) -> dict:
    """Best-effort license extraction from core metadata (PEP 639 aware)."""
    meta = dist.metadata
    expression = (meta.get("License-Expression") or "").strip()
    field = (meta.get("License") or "").strip()
    classifiers = [c for c in (meta.get_all("Classifier") or []) if c.startswith("License ::")]
    return {
        "license_expression": expression or None,
        "license_field": field or None,
        "license_classifiers": classifiers,
    }


def effective_license(info: dict) -> str:
    """Prefer PEP 639 expression, then the License field, then classifiers."""
    return (
        info["license_expression"]
        or info["license_field"]
        or "; ".join(info["license_classifiers"])
    )


def truncate(text: str, limit: int = 160) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def classify(license_text: str) -> dict:
    """Bucket a license string per the G15 policy and derive its status."""
    text = license_text.strip().lower()
    if not text:
        return {"bucket": "unknown", "status": "REQUIRES_LEGAL_REVIEW"}
    for marker in COPYLEFT_MARKERS:
        if re.search(marker, text):
            return {"bucket": "copyleft", "status": "REQUIRES_LEGAL_REVIEW"}
    for marker in PERMISSIVE_MARKERS:
        if re.search(marker, text):
            return {"bucket": "permissive", "status": "CLEARED"}
    return {"bucket": "other", "status": "REQUIRES_LEGAL_REVIEW"}


def satisfies(version: str, requirement: str):
    """True/False when 'packaging' is importable, else None."""
    try:
        from packaging.requirements import Requirement

        return Requirement(requirement).specifier.contains(version, prereleases=True)
    except Exception:
        return None


def installed_distributions() -> dict:
    """Canonical name -> distribution, de-duplicated."""
    found = {}
    for dist in md.distributions():
        try:
            raw_name = dist.metadata["Name"]
            version = dist.version
        except Exception:
            continue
        if not raw_name:
            continue
        name = canonical(raw_name)
        info = dist_license(dist)
        prior = found.get(name)
        if prior is None or (not effective_license(prior[1]) and effective_license(info)):
            found[name] = (dist, info, version)
    return found


def main() -> int:
    with PYPROJECT.open("rb") as handle:
        pyproject = tomllib.load(handle)

    project = pyproject["project"]
    runtime_specs = list(project["dependencies"])
    optional_specs = {
        extra: list(reqs)
        for extra, reqs in project.get("optional-dependencies", {}).items()
    }
    project_name = canonical(project["name"])

    runtime_names = {canonical(requirement_name(spec)): spec for spec in runtime_specs}
    optional_names = {}
    for extra, specs in optional_specs.items():
        for spec in specs:
            optional_names[canonical(requirement_name(spec))] = (extra, spec)

    installed = installed_distributions()

    runtime_report = []
    for spec in runtime_specs:
        name = canonical(requirement_name(spec))
        entry = {
            "package": name,
            "pyproject_spec": spec,
            "installed": name in installed,
        }
        if name in installed:
            dist, info, version = installed[name]
            license_text = effective_license(info)
            verdict = classify(license_text)
            ok = satisfies(version, spec)
            entry.update(
                {
                    "installed_name": dist.metadata["Name"],
                    "version": version,
                    "version_satisfies_spec": ok,
                    "license": license_text or None,
                    "license_expression": info["license_expression"],
                    "license_field": truncate(info["license_field"] or ""),
                    "license_classifiers": info["license_classifiers"],
                    "bucket": verdict["bucket"],
                    "status": verdict["status"],
                    "notes": (
                        "license from %s"
                        % (
                            "License-Expression metadata"
                            if info["license_expression"]
                            else "License metadata field"
                            if info["license_field"]
                            else "License classifiers"
                        )
                    ),
                }
            )
            if verdict["bucket"] == "permissive":
                entry["notes"] += "; in the pre-cleared permissive set (MIT/BSD/Apache/ISC)"
            elif verdict["bucket"] == "unknown":
                entry["notes"] += "; no license metadata found"
            else:
                entry["notes"] += "; outside the pre-cleared permissive set"
        else:
            entry.update(
                {
                    "version": None,
                    "license": None,
                    "bucket": "unknown",
                    "status": "REQUIRES_LEGAL_REVIEW",
                    "notes": "declared in pyproject.toml but not installed in the scanned environment",
                }
            )
        runtime_report.append(entry)

    excluded = []
    for name, (dist, info, version) in sorted(installed.items()):
        if name in runtime_names:
            continue
        if name == project_name:
            reason = "the project itself"
        elif name in optional_names:
            reason = "optional extra [project.optional-dependencies].%s (%s); not core runtime" % (
                optional_names[name][0],
                optional_names[name][1],
            )
        else:
            reason = "not declared in [project.dependencies] (dev/test/other)"
        license_text = effective_license(info)
        verdict = classify(license_text)
        excluded.append(
            {
                "package": name,
                "version": version,
                "license": truncate(license_text) or None,
                "bucket": verdict["bucket"],
                # Outside the shipped runtime set by definition -> NOT_APPLICABLE.
                "status": "NOT_APPLICABLE",
                "reason": reason,
            }
        )

    buckets = {}
    for entry in runtime_report:
        buckets[entry["bucket"]] = buckets.get(entry["bucket"], 0) + 1

    report = {
        "format": "veya-license-scan-v1",
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "pyproject": str(PYPROJECT.relative_to(ROOT)),
        "runtime_requirement_specs": runtime_specs,
        "optional_extra_specs": optional_specs,
        "runtime_dependencies": runtime_report,
        "runtime_buckets": buckets,
        "excluded_installed_packages": excluded,
        "excluded_count": len(excluded),
    }
    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
