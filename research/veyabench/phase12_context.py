"""Causal SGD USER-turn inputs and explicit context packing for Phase 12.

Official test dialogues are deliberately inaccessible from this module. A
separate sealed harness may read them only after the finalist is frozen.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any

from research.veyabench.phase12_sgd_inventory import SOURCE, file_hash
from veya.common import build_sequence


MODES = ("utterance_only", "previous_turn", "full_context")


def load_examples(split: str, services: set[str], source_hashes: dict[str, str]) -> list[dict]:
    if split not in ("train", "dev"):
        raise ValueError("test dialogues require the sealed finalist harness")
    rows = []
    for raw_path in sorted(glob.glob(str(SOURCE / split / "dialogues_*.json"))):
        path = Path(raw_path)
        relative = str(path.relative_to(SOURCE))
        if file_hash(path) != source_hashes[relative]:
            raise ValueError(f"SGD file changed: {relative}")
        with path.open(encoding="utf-8") as handle:
            dialogues = json.load(handle)
        rows.extend(examples_from_dialogues(dialogues, split, services))
    return rows


def examples_from_dialogues(dialogues: list[dict], split: str, services: set[str]) -> list[dict]:
    rows = []
    for dialogue in dialogues:
        history: list[tuple[str, str]] = []
        previous_user_services: set[str] = set()
        for turn_index, turn in enumerate(dialogue["turns"]):
            speaker = turn["speaker"]
            utterance = turn["utterance"]
            if speaker not in ("USER", "SYSTEM"):
                raise ValueError("unknown SGD speaker")
            if speaker == "USER":
                current_services = {frame["service"] for frame in turn["frames"]}
                for frame_index, frame in enumerate(turn["frames"]):
                    service = frame["service"]
                    if service not in services:
                        continue
                    rows.append({
                        "id": f"{split}:{dialogue['dialogue_id']}:{turn_index}:{frame_index}",
                        "service": service,
                        "gold": frame["state"]["active_intent"],
                        "current": utterance,
                        "history": tuple(history),
                        "service_transition": bool(previous_user_services and service not in previous_user_services),
                        "initial_user_turn": not any(item[0] == "USER" for item in history),
                    })
                previous_user_services = current_services
            history.append((speaker, utterance))
    return rows


def turns_for(example: dict, mode: str) -> list[tuple[str, str]]:
    if mode not in MODES:
        raise ValueError(f"unknown context mode {mode}")
    history = example["history"]
    if mode == "utterance_only":
        prior = []
    elif mode == "previous_turn":
        prior = list(history[-1:])
    else:
        prior = list(history)
    return prior + [("USER", example["current"])]


def render_turns(turns: list[tuple[str, str]]) -> str:
    # Keep each annotated speaker on one physical line. Embedded line breaks in
    # an utterance must not be able to masquerade as another dialogue turn.
    lines = []
    for speaker, utterance in turns:
        escaped = utterance.strip().replace("\r", "\\r").replace("\n", "\\n")
        lines.append(f"{speaker}: {escaped}")
    return "\n".join(lines)


def pack_context(
    tokenizer: Any,
    example: dict,
    mode: str,
    question: dict,
    *,
    max_len: int,
    head_max_len: int,
) -> tuple[str | None, dict]:
    """Keep the current turn and newest whole prior turns within the exact head budget."""
    turns = turns_for(example, mode)
    if not example["current"].strip():
        return None, {"reason": "empty_current_turn"}
    header, markers = build_sequence(tokenizer, "", question, max_len, head_max_len)
    if len(markers) != len(question["crit"]):
        raise ValueError("candidate head exceeds model limit")
    room = max_len - len(header)

    def token_count(value: str) -> int:
        return len(tokenizer(value.replace(tokenizer.mask_token, " "), add_special_tokens=False)["input_ids"])

    current = [turns[-1]]
    current_text = render_turns(current)
    current_tokens = token_count(current_text)
    raw_text = render_turns(turns)
    raw_tokens = token_count(raw_text)
    if current_tokens > room:
        return None, {
            "reason": "current_turn_over_budget",
            "state_token_budget": room,
            "current_tokens": current_tokens,
            "raw_tokens": raw_tokens,
            "raw_turns": len(turns),
        }
    kept = current
    for prior in reversed(turns[:-1]):
        candidate = [prior] + kept
        if token_count(render_turns(candidate)) > room:
            break
        kept = candidate
    packed = render_turns(kept)
    packed_tokens = token_count(packed)
    sequence, retained_markers = build_sequence(tokenizer, packed, question, max_len, head_max_len)
    if len(sequence) != len(header) + packed_tokens or len(retained_markers) != len(markers):
        raise ValueError("model would silently truncate packed context")
    if not packed.endswith(current_text):
        raise ValueError("current USER turn was not retained")
    return packed, {
        "reason": None,
        "state_token_budget": room,
        "current_tokens": current_tokens,
        "raw_tokens": raw_tokens,
        "packed_tokens": packed_tokens,
        "raw_turns": len(turns),
        "packed_turns": len(kept),
        "dropped_prior_turns": len(turns) - len(kept),
    }
