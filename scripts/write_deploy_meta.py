"""Write deploy identity from git and the audited release manifest.

Run this before `wrangler deploy`. The committed source-commit string is not
the deploy identity: this script overwrites worker/build-meta.mjs from the
checked-out revision. dirty is true when any path that ships is dirty.
Unrelated research edits do not count. The manifest file hash and the
checkpoint hash are checked, not copied by hand into a second place.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "artifacts" / "release" / "manifest.json"
CHECKPOINT_PATH = ROOT / "models" / "production" / "model.safetensors"
# wrangler.jsonc is the plain `veya` edge proxy that Workers Builds deploys from
# main; it carries no release pins. The pinned Container Workers each have their
# own config, and every one of them must agree with the audited manifest.
PINNED_CONFIG_PATHS = (
    ROOT / "wrangler.staging.jsonc",
    ROOT / "wrangler.production.jsonc",
)
CONFIG_PATH = PINNED_CONFIG_PATHS[1]
BUILD_META_PATH = ROOT / "worker" / "build-meta.mjs"
PINS_PATH = ROOT / "deploy" / "cloudflare" / "release-pins.json"

AUDITED_RELEASE = "0.3.5-audit-baseline.1"
AUDITED_CHECKPOINT_SHA256 = "891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c"
AUDITED_CHECKPOINT_BYTES = 842609210
AUDITED_MANIFEST_SHA256 = "80720058c6b07dfef0640171d4193ff7bcaa3f40629ac60df7f9b28e074d65a4"

SHIP = (
    "veya",
    "web",
    "worker",
    "deploy/cloudflare",
    "wrangler.staging.jsonc",
    "wrangler.production.jsonc",
    "artifacts/release/manifest.json",
    "site",
    "package.json",
    "package-lock.json",
    "scripts/write_deploy_meta.py",
    ".dockerignore",
)
GENERATED = {
    "worker/build-meta.mjs",
    "deploy/cloudflare/release-pins.json",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strip_jsonc(text: str) -> str:
    out: list[str] = []
    in_string = False
    escape = False
    index = 0
    while index < len(text):
        char = text[index]
        if in_string:
            out.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            out.append(char)
            index += 1
            continue
        if char == "/" and text[index : index + 2] == "//":
            newline = text.find("\n", index)
            if newline < 0:
                break
            index = newline
            continue
        out.append(char)
        index += 1
    return "".join(out)


def verify_release(
    manifest_path: Path = MANIFEST_PATH,
    checkpoint_path: Path | None = CHECKPOINT_PATH,
) -> dict[str, object]:
    raw = manifest_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != AUDITED_MANIFEST_SHA256:
        raise SystemExit("manifest bytes do not match the audited digest")
    data = json.loads(raw)
    if data.get("release") != AUDITED_RELEASE:
        raise SystemExit("release id does not match the audited manifest")
    if data.get("base_checkpoint_hash") != AUDITED_CHECKPOINT_SHA256:
        raise SystemExit("manifest checkpoint hash does not match the audited checkpoint")
    if checkpoint_path is not None and checkpoint_path.is_file() and not checkpoint_path.is_symlink():
        size = checkpoint_path.stat().st_size
        actual = _sha256(checkpoint_path)
        if size != AUDITED_CHECKPOINT_BYTES or actual != AUDITED_CHECKPOINT_SHA256:
            raise SystemExit("local checkpoint does not match the audited manifest")
    return data


def assert_config_pins(config: dict[str, object]) -> None:
    variables = config.get("vars")
    if not isinstance(variables, dict):
        raise SystemExit("wrangler vars are missing")
    if variables.get("VEYA_RELEASE_ID") != AUDITED_RELEASE:
        raise SystemExit("wrangler release id does not match the manifest")
    if variables.get("VEYA_CHECKPOINT_SHA256") != AUDITED_CHECKPOINT_SHA256:
        raise SystemExit("wrangler checkpoint sha does not match the manifest")
    if variables.get("VEYA_MANIFEST_SHA256") != AUDITED_MANIFEST_SHA256:
        raise SystemExit("wrangler manifest sha does not match the manifest bytes")
    if "VEYA_SOURCE_COMMIT" in variables or "VEYA_SOURCE_DIRTY" in variables:
        raise SystemExit("source identity is generated, not stored in wrangler vars")


def identity_from_status(commit: str, porcelain: str) -> tuple[str, str]:
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise SystemExit("source commit is not a full git revision")
    dirty = False
    for line in porcelain.splitlines():
        if len(line) < 4:
            continue
        path = line[3:].strip().strip('"')
        if " -> " in path:
            path = path.split(" -> ", 1)[1].strip().strip('"')
        if path in GENERATED:
            continue
        dirty = True
        break
    return commit, "true" if dirty else "false"


def source_identity(root: Path = ROOT) -> tuple[str, str]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    porcelain = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *SHIP], cwd=root, text=True
    )
    return identity_from_status(commit, porcelain)


def pins() -> dict[str, str]:
    return {
        "VEYA_RELEASE_ID": AUDITED_RELEASE,
        "VEYA_CHECKPOINT_SHA256": AUDITED_CHECKPOINT_SHA256,
        "VEYA_CHECKPOINT_BYTES": str(AUDITED_CHECKPOINT_BYTES),
        "VEYA_MANIFEST_SHA256": AUDITED_MANIFEST_SHA256,
    }


def render_build_meta(commit: str, dirty: str) -> str:
    values = pins()
    return (
        "// Generated by scripts/write_deploy_meta.py. Regenerate before wrangler deploy.\n"
        f'export const RELEASE_ID = "{values["VEYA_RELEASE_ID"]}";\n'
        f'export const CHECKPOINT_SHA256 = "{values["VEYA_CHECKPOINT_SHA256"]}";\n'
        f'export const CHECKPOINT_BYTES = "{values["VEYA_CHECKPOINT_BYTES"]}";\n'
        f'export const MANIFEST_SHA256 = "{values["VEYA_MANIFEST_SHA256"]}";\n'
        f'export const SOURCE_COMMIT = "{commit}";\n'
        f'export const SOURCE_DIRTY = "{dirty}";\n'
    )


def main() -> None:
    verify_release()
    for path in PINNED_CONFIG_PATHS:
        config = json.loads(strip_jsonc(path.read_text(encoding="utf-8")))
        assert_config_pins(config)
    commit, dirty = source_identity()
    BUILD_META_PATH.write_text(render_build_meta(commit, dirty), encoding="utf-8")
    PINS_PATH.write_text(json.dumps(pins(), indent=2) + "\n", encoding="utf-8")
    print(f"source_commit={commit} source_dirty={dirty}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit as exc:
        if exc.code not in (None, 0):
            print(exc.code, file=sys.stderr)
        raise
