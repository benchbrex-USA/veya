#!/usr/bin/env python3
"""Synchronize Veya Foundry research sources into an ignored workspace.

The committed Veya package never imports these sources directly. Each source is
kept isolated so Codex can inspect, benchmark, and bridge it without polluting
Veya's runtime dependencies or namespace.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def run(cmd: list[str], cwd: Path | None = None) -> str:
    proc = subprocess.run(cmd, cwd=cwd, check=True, text=True, capture_output=True)
    return proc.stdout.strip()


def sync_source(root: Path, source: dict, include_reference_only: bool) -> dict:
    source_id = source["source_id"]
    policy = source.get("integration_policy", "reference_only")
    if policy != "integrate" and not include_reference_only:
        return {"source_id": source_id, "status": "skipped", "reason": policy}

    dest = root / "sources" / source_id
    repo = source["repository"]
    branch = source.get("branch") or "main"
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--depth", "1", "--branch", branch, repo, str(dest)])
    else:
        run(["git", "fetch", "--depth", "1", "origin", branch], cwd=dest)
        run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=dest)
    sha = run(["git", "rev-parse", "HEAD"], cwd=dest)
    return {
        "source_id": source_id,
        "status": "synced",
        "commit": sha,
        "integration_policy": policy,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="research/provenance/sources.json")
    parser.add_argument("--workspace", default=".veya-foundry")
    parser.add_argument("--include-reference-only", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = Path(args.workspace)
    root.mkdir(parents=True, exist_ok=True)
    results = [
        sync_source(root, source, args.include_reference_only)
        for source in payload["sources"]
    ]
    lock_path = root / "sources.lock.json"
    lock_path.write_text(json.dumps({"sources": results}, indent=2) + "\n", encoding="utf-8")
    print(lock_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
