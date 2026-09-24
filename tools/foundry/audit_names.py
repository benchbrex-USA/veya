#!/usr/bin/env python3
"""Enforce the Veya-only first-party naming boundary.

Research provenance and ignored source snapshots are intentionally excluded:
those locations must preserve upstream names for legal and scientific traceability.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TEXT_SUFFIXES = {
    ".py", ".toml", ".yaml", ".yml", ".json", ".md", ".txt", ".ini", ".cfg", ".sh"
}


def load_terms(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    terms = []
    for source in payload["sources"]:
        terms.extend(source.get("audit_terms") or [])
    return sorted({term.lower() for term in terms if term})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="veya")
    parser.add_argument("--manifest", default="research/provenance/sources.json")
    args = parser.parse_args()

    root = Path(args.root)
    terms = load_terms(Path(args.manifest))
    violations = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8").lower()
        except UnicodeDecodeError:
            continue
        for term in terms:
            if term in text:
                violations.append((str(path), term))
    if violations:
        for path, term in violations:
            print(f"{path}: contains forbidden upstream term {term!r}")
        return 1
    print("Veya naming audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
