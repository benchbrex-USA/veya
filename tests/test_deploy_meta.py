"""Deploy metadata is checked against the manifest, not copied by hand."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import write_deploy_meta as meta  # noqa: E402


def test_audited_manifest_and_wrangler_pins_agree():
    data = meta.verify_release()
    assert data["release"] == meta.AUDITED_RELEASE
    assert data["base_checkpoint_hash"] == meta.AUDITED_CHECKPOINT_SHA256
    config = json.loads(meta.strip_jsonc((ROOT / "wrangler.jsonc").read_text(encoding="utf-8")))
    meta.assert_config_pins(config)


def test_dirty_is_false_only_when_shipped_paths_are_clean():
    commit = "a" * 40
    assert meta.identity_from_status(commit, "") == (commit, "false")
    assert meta.identity_from_status(commit, " M web/server.py") == (commit, "true")
    assert meta.identity_from_status(commit, "?? worker/build-meta.mjs") == (commit, "false")
    assert all(not item.startswith("research") for item in meta.SHIP)


def test_build_meta_carries_the_verified_hashes():
    text = meta.render_build_meta("b" * 40, "false")
    assert meta.AUDITED_CHECKPOINT_SHA256 in text
    assert meta.AUDITED_MANIFEST_SHA256 in text
    assert 'SOURCE_DIRTY = "false"' in text
    pins = meta.pins()
    assert pins["VEYA_CHECKPOINT_BYTES"] == str(meta.AUDITED_CHECKPOINT_BYTES)
