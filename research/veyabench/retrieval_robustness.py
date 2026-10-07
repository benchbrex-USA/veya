"""Label-preserving query noise and deterministic views.

Views are derived only from the string the retriever would see. They do not
receive the original clean query as a side channel.
"""

from __future__ import annotations

import json

SHORT_SUFFIX = (
    "\nReference: ticket 1842. Order status: pending. Timestamp: 2026-01-04T09:00:00Z."
)
LONG_BOILERPLATE = (
    " Unrelated operations note. The warehouse shift is complete and the lobby "
    "lights were reset. This sentence does not describe the customer request."
)
PREFIX = "System note: customer tier gold. Channel: chat. Attempt: 2.\n"


def apply_perturbation(text: str, kind: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    if kind == "clean":
        return text
    if kind == "short_suffix":
        return text + SHORT_SUFFIX
    if kind == "long_suffix":
        return text + SHORT_SUFFIX + (LONG_BOILERPLATE * 12)
    if kind == "prefix":
        return PREFIX + text
    if kind == "json_wrapper":
        return json.dumps(
            {
                "ticket": "1842",
                "channel": "chat",
                "timestamp": "2026-01-04T09:00:00Z",
                "text": text,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    raise ValueError("unknown perturbation %r" % kind)


def adversarial_suffix(wrong_naturalized: str) -> str:
    """Data suffix naming one incorrect intent. Not an instruction."""
    if not isinstance(wrong_naturalized, str) or not wrong_naturalized.strip():
        raise ValueError("wrong intent text is required")
    if "\n" in wrong_naturalized or len(wrong_naturalized) > 200:
        raise ValueError("wrong intent text is not a single short label")
    return "\nMetadata category: %s." % wrong_naturalized.strip()


def choose_wrong_index(ranked: list[int] | tuple[int, ...], gold_index: int, naturalized: list[str]) -> int:
    """First ranked candidate whose text differs from the gold label."""
    gold_text = naturalized[gold_index]
    for index in ranked:
        index = int(index)
        if index == gold_index or naturalized[index] == gold_text:
            continue
        return index
    raise ValueError("no distinct wrong candidate")


# Fixed before the end-to-end run. Do not retune these after seeing scores.
CLEAN_E2E_REGRESSION_LIMIT = 0.01
ADVERSARIAL_HARM_MARGIN = 0.05
ADVERSARIAL_TOP1_MARGIN = 0.10


def classify_outcome(
    *,
    baseline_correct: bool,
    fusion_correct: bool,
    baseline_hit: bool,
    fusion_hit: bool,
    adversarial: bool,
    prediction: str,
    injected: str | None,
) -> dict[str, object]:
    recovery = fusion_hit and not baseline_hit
    loss = baseline_hit and not fusion_hit
    capture = bool(
        adversarial
        and injected
        and prediction == injected
        and baseline_correct
        and not fusion_correct
    )
    if baseline_correct and fusion_correct:
        primary = "UNCHANGED_CORRECT"
    elif (not baseline_correct) and (not fusion_correct):
        primary = "UNCHANGED_WRONG"
    elif fusion_correct:
        primary = "GOOD_FLIP"
    else:
        primary = "BAD_FLIP"
    return {
        "primary": primary,
        "retrieval_recovery": recovery,
        "retrieval_loss": loss,
        "adversarial_capture": capture,
    }


def fusion_recommendation(stats: dict) -> str:
    """Map predeclared gates to REJECT, RESEARCH, or CANDIDATE. Never PROMOTED."""
    benign = stats["short_e2e_delta"] > 0 and stats["long_e2e_delta"] > 0
    clean_ok = stats["clean_e2e_delta"] >= -CLEAN_E2E_REGRESSION_LIMIT
    long_retrieval = stats["long_recall8_delta_vs_phase10_full"] > 0
    adversarial_worse = (
        stats["fusion_harm_rate"] > stats["production_harm_rate"] + ADVERSARIAL_HARM_MARGIN
        or stats["fusion_wrong_top1"] > stats["production_wrong_top1"] + ADVERSARIAL_TOP1_MARGIN
    )
    both_benign_failed = stats["short_e2e_delta"] <= 0 and stats["long_e2e_delta"] <= 0
    if both_benign_failed and adversarial_worse:
        return "REJECT"
    if benign and clean_ok and long_retrieval and not adversarial_worse:
        return "CANDIDATE"
    if benign and adversarial_worse:
        return "RESEARCH"
    if both_benign_failed:
        return "REJECT"
    return "RESEARCH"


def query_views(text: str) -> dict[str, str]:
    """Bounded views of one observed string. The label is not an input."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    views = {
        "full": text.strip()[:4096],
        "head": lines[0][:512],
        "tail": lines[-1][:512],
    }
    parsed = _json_text_field(text)
    if parsed:
        views["json_text"] = parsed[:4096]
    return views


def _json_text_field(text: str) -> str | None:
    stripped = text.strip()
    if not stripped.startswith("{"):
        return None
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    value = payload.get("text") if isinstance(payload, dict) else None
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None
