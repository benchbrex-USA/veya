"""Fresh-process production load. Run as a script, not at import time."""

from __future__ import annotations

import json
import os
import resource
import time

from veya import __version__
from veya.release_bundle import load_release_manifest


def main() -> dict:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    manifest_path = os.path.join(root, "artifacts/release/manifest.json")
    started = time.monotonic()
    bundle, manifest = load_release_manifest(manifest_path, device="cpu", root=root)
    load_s = time.monotonic() - started
    questions = {
        "team": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {"billing": "invoices and refunds", "technical": "bugs and outages"},
        },
        "cancel": {"type": "noul", "instructions": "Does the user ask to cancel?"},
        "severity": {
            "type": "score",
            "instructions": "How severe is this?",
            "criteria": ["nominal", "minor", "serious", "critical"],
        },
    }
    infer_started = time.monotonic()
    result = bundle.answer_questions("Please refund the duplicate March invoice.", questions)
    infer_s = time.monotonic() - infer_started
    answers = result.get("answers") or {}
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "measurement": "measured",
        "success": result.get("decision") == "answer" and all(qid in answers for qid in questions),
        "load_s": round(load_s, 3),
        "infer_s": round(infer_s, 3),
        "rss_bytes": int(rss),
        "device": "cpu",
        "veya_version": __version__,
        "release": manifest.get("release"),
        "manifest_format": manifest.get("format"),
        "answers": {
            "team": (answers.get("team") or {}).get("choice"),
            "cancel": (answers.get("cancel") or {}).get("noul"),
            "severity": (answers.get("severity") or {}).get("score"),
        },
        "checkpoint_sha256": bundle.checkpoint_hash,
        "generalization_claim": False,
    }


if __name__ == "__main__":
    print(json.dumps(main(), sort_keys=True))
