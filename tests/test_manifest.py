"""Regression tests for the checkpoint manifest builder (veya/manifest.py).

Pure file hashing and config reading: no model load, no network.

Run: python tests/test_manifest.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from veya.manifest import build_manifest  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name if cond else "%s %s" % (name, detail))


def _make_checkpoint(directory: Path, encoder: str, model_name=None, with_calibration=False):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "rl_agent_config.json").write_text(json.dumps({
        "encoder": encoder, "head_layers": 2, "max_len": 512, "head_max_len": 192,
        **({"model_name": model_name} if model_name else {}),
    }))
    torch.save({"weight": torch.zeros(1)}, directory / "model.safetensors")
    if with_calibration:
        (directory / "calibration.json").write_text("{}")


def test_manifest_fields():
    import hashlib
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        directory = root / "ckpt"
        _make_checkpoint(directory, "modernbert-large", model_name="veya", with_calibration=True)
        manifest = build_manifest(str(directory), git_commit="abc123")
        check("manifest/model_id from cfg", manifest["model_id"] == "veya", manifest["model_id"])
        check("manifest/backbone", manifest["backbone"] == "modernbert-large")
        check("manifest/architecture version", manifest["architecture_version"] == "0.3.5")
        expected = hashlib.sha256((directory / "model.safetensors").read_bytes()).hexdigest()
        check("manifest/checkpoint hash", manifest["checkpoint_hash"] == expected)
        check("manifest/calibration artifact", manifest["calibration_artifact"] == "calibration.json")
        check("manifest/languages latin", manifest["supported_languages"] == "latin")
        check("manifest/max state tokens", manifest["max_state_tokens"] == 512)
        check("manifest/max candidates packed", manifest["max_candidates_packed"] == 192)
        check("manifest/primitives", manifest["decision_primitives"] == ["choice", "score", "noul"])
        check("manifest/git commit", manifest["git_commit"] == "abc123")
        check("manifest/created_at shape", len(manifest["created_at"]) == 20 and manifest["created_at"].endswith("Z"))


def test_manifest_defaults_and_multilingual():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        directory = root / "mmbert-ckpt"
        _make_checkpoint(directory, "mmbert-base")
        manifest = build_manifest(str(directory))
        check("manifest/model_id falls back to dir name", manifest["model_id"] == "mmbert-ckpt")
        check("manifest/multilingual detected", manifest["supported_languages"] == "multilingual")
        check("manifest/no calibration -> None", manifest["calibration_artifact"] is None)
        check("manifest/git_commit None", manifest["git_commit"] is None)


def test_manifest_missing_weights_gives_null_hash():
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp) / "bare"
        directory.mkdir()
        (directory / "rl_agent_config.json").write_text(json.dumps({"encoder": "e", "head_layers": 2}))
        manifest = build_manifest(str(directory))
        check("manifest/missing weights -> None", manifest["checkpoint_hash"] is None)


def test_recorded_checks_pass():
    assert not FAIL, "\n".join(FAIL)


if __name__ == "__main__":
    test_manifest_fields()
    test_manifest_defaults_and_multilingual()
    test_manifest_missing_weights_gives_null_hash()
    print("%d passed, %d failed" % (len(PASS), len(FAIL)))
    for item in FAIL:
        print(" FAIL", item)
    raise SystemExit(1 if FAIL else 0)
