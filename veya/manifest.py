"""Checkpoint manifest. Hashes the weight file the runtime actually loads."""
import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Optional




def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(model_dir: str, git_commit: Optional[str] = None) -> dict:
    config_path = os.path.join(model_dir, "rl_agent_config.json")
    weights = os.path.join(model_dir, "model.safetensors")
    with open(config_path, encoding="utf-8") as handle:
        cfg = json.load(handle)
    import veya
    encoder = str(cfg.get("encoder", ""))
    multilingual = "mmbert" in encoder.lower()
    return {
        "model_id": cfg.get("model_name") or os.path.basename(os.path.abspath(model_dir)),
        "architecture_version": veya.__version__,
        "backbone": encoder,
        "checkpoint_hash": _sha256(weights) if os.path.exists(weights) else None,
        "calibration_artifact": "calibration.json" if os.path.exists(os.path.join(model_dir, "calibration.json")) else None,
        "supported_languages": "multilingual" if multilingual else "latin",
        "max_state_tokens": cfg.get("max_len"),
        "max_candidates_packed": cfg.get("head_max_len"),
        "decision_primitives": ["choice", "score", "noul"],
        "git_commit": git_commit,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
