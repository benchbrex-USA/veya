"""Fail when a CodeQL SARIF file contains error- or warning-level results.

SARIF results omit `level` when it equals the rule default; the SARIF default
level is "warning". Suppressed results (e.g. `# lgtm` / `codeql[...]` inline
suppressions) are ignored.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

FAILING = {"error", "warning"}
ALLOWLIST = Path(__file__).resolve().parents[1] / "codeql" / "allowlist.json"


def load_allowlist(path: Path = ALLOWLIST) -> set[tuple[str, str]]:
    if not path.is_file():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    allowed = set()
    for entry in data.get("entries", []):
        if not entry.get("reason"):
            raise SystemExit(f"allowlist entry without a reason: {entry}")
        allowed.add((entry["rule"], entry["path"]))
    return allowed


def rule_levels(run: dict) -> dict[str, str]:
    levels: dict[str, str] = {}
    driver = run.get("tool", {}).get("driver", {})
    extensions = run.get("tool", {}).get("extensions", []) or []
    for component in [driver, *extensions]:
        for rule in component.get("rules", []) or []:
            level = (rule.get("defaultConfiguration") or {}).get("level")
            if rule.get("id") and level:
                levels[rule["id"]] = level
    return levels


def findings(sarif: dict, allowed: set[tuple[str, str]] | None = None) -> list[tuple[str, str, str, str]]:
    allowed = allowed or set()
    out = []
    for run in sarif.get("runs", []) or []:
        levels = rule_levels(run)
        for result in run.get("results", []) or []:
            if any(item.get("kind") != "inSource" or item.get("status") == "accepted" for item in result.get("suppressions", []) or []):
                continue
            rule = result.get("ruleId") or (result.get("rule") or {}).get("id") or "?"
            level = result.get("level") or levels.get(rule) or "warning"
            if level not in FAILING:
                continue
            where = "?"
            uri = "?"
            locations = result.get("locations") or []
            if locations:
                physical = locations[0].get("physicalLocation", {})
                uri = physical.get("artifactLocation", {}).get("uri", "?")
                line = physical.get("region", {}).get("startLine", "?")
                where = f"{uri}:{line}"
            if (rule, uri) in allowed:
                print(f"allowlisted: {rule} {where}")
                continue
            message = (result.get("message") or {}).get("text", "").splitlines()[0][:200]
            out.append((level, rule, where, message))
    return out


def main(argv: list[str]) -> int:
    root = Path(argv[1] if len(argv) > 1 else "codeql-results")
    files = sorted(root.rglob("*.sarif")) if root.is_dir() else [root]
    if not files:
        print(f"::error::no SARIF files under {root}")
        return 1
    allowed = load_allowlist()
    total = []
    for path in files:
        total.extend(findings(json.loads(path.read_text(encoding="utf-8")), allowed))
    for level, rule, where, message in total:
        print(f"::{'error' if level == 'error' else 'warning'} title=CodeQL {rule}::{where} {message}")
    if total:
        print(f"CodeQL gate: {len(total)} error/warning finding(s) in {len(files)} SARIF file(s)")
        return 1
    print(f"CodeQL gate: 0 error/warning findings in {len(files)} SARIF file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
