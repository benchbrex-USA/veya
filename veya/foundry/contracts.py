"""Stable Veya Foundry request/response contracts.

The Foundry intentionally keeps source-specific implementation details behind a
process boundary. First-party Veya code only depends on these types.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Tuple


class VeyaQuestionType(str, Enum):
    CHOICE = "choice"
    SCORE = "score"
    BOOLEAN = "boolean"


@dataclass(frozen=True)
class VeyaQuestion:
    """One typed decision question supplied at runtime."""

    question_id: str
    question_type: VeyaQuestionType
    instruction: str
    options: Tuple[str, ...] = ()
    option_descriptions: Mapping[str, Optional[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.question_id.strip():
            raise ValueError("question_id must not be empty")
        if not self.instruction.strip():
            raise ValueError("instruction must not be empty")
        if self.question_type is VeyaQuestionType.BOOLEAN:
            if self.options:
                raise ValueError("boolean questions do not accept explicit options")
            return
        if len(self.options) < 1:
            raise ValueError(f"{self.question_type.value} questions require options")
        if len(set(self.options)) != len(self.options):
            raise ValueError("question options must be unique")
        unknown = set(self.option_descriptions) - set(self.options)
        if unknown:
            raise ValueError(f"descriptions supplied for unknown options: {sorted(unknown)!r}")

    def normalized_options(self) -> Tuple[str, ...]:
        if self.question_type is VeyaQuestionType.BOOLEAN:
            return ("false", "true")
        return self.options


@dataclass(frozen=True)
class VeyaRequest:
    """A Veya decision request shared by every Foundry expert."""

    state: Any
    questions: Tuple[VeyaQuestion, ...]
    request_id: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.questions:
            raise ValueError("at least one question is required")
        ids = [q.question_id for q in self.questions]
        if len(ids) != len(set(ids)):
            raise ValueError("question ids must be unique within a request")


@dataclass(frozen=True)
class VeyaDistribution:
    """Normalized probability distribution for one question."""

    labels: Tuple[str, ...]
    probabilities: Tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.labels:
            raise ValueError("distribution requires at least one label")
        if len(self.labels) != len(self.probabilities):
            raise ValueError("labels and probabilities must have the same length")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("distribution labels must be unique")
        total = 0.0
        for p in self.probabilities:
            if p < 0.0 or p > 1.0:
                raise ValueError("probabilities must be within [0, 1]")
            total += p
        if abs(total - 1.0) > 1e-5:
            raise ValueError(f"probabilities must sum to 1.0, got {total}")

    @classmethod
    def from_mapping(cls, scores: Mapping[str, float]) -> "VeyaDistribution":
        labels = tuple(scores)
        probs = tuple(float(scores[label]) for label in labels)
        total = sum(probs)
        if total <= 0:
            raise ValueError("score sum must be positive")
        return cls(labels=labels, probabilities=tuple(p / total for p in probs))

    def as_mapping(self) -> Dict[str, float]:
        return dict(zip(self.labels, self.probabilities))

    @property
    def selected(self) -> str:
        index = max(range(len(self.probabilities)), key=self.probabilities.__getitem__)
        return self.labels[index]

    @property
    def confidence(self) -> float:
        return max(self.probabilities)


@dataclass(frozen=True)
class VeyaAnswer:
    question_id: str
    distribution: VeyaDistribution
    expected_score: Optional[float] = None

    @property
    def selected(self) -> str:
        return self.distribution.selected


@dataclass(frozen=True)
class VeyaResponse:
    expert_id: str
    answers: Mapping[str, VeyaAnswer]
    latency_ms: Optional[float] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.expert_id.strip():
            raise ValueError("expert_id must not be empty")
        for key, answer in self.answers.items():
            if key != answer.question_id:
                raise ValueError("answer map key must match answer.question_id")


def request_to_dict(request: VeyaRequest) -> Dict[str, Any]:
    return {
        "request_id": request.request_id,
        "state": request.state,
        "metadata": dict(request.metadata),
        "questions": [
            {
                "id": q.question_id,
                "type": q.question_type.value,
                "instruction": q.instruction,
                "options": list(q.options),
                "descriptions": dict(q.option_descriptions),
            }
            for q in request.questions
        ],
    }


def response_from_dict(payload: Mapping[str, Any]) -> VeyaResponse:
    expert_id = str(payload["expert_id"])
    answers: Dict[str, VeyaAnswer] = {}
    raw_answers = payload.get("answers") or {}
    for question_id, raw in raw_answers.items():
        distribution = VeyaDistribution.from_mapping(raw["probabilities"])
        answers[str(question_id)] = VeyaAnswer(
            question_id=str(question_id),
            distribution=distribution,
            expected_score=(None if raw.get("expected_score") is None else float(raw["expected_score"])),
        )
    latency = payload.get("latency_ms")
    return VeyaResponse(
        expert_id=expert_id,
        answers=answers,
        latency_ms=None if latency is None else float(latency),
        metadata=dict(payload.get("metadata") or {}),
    )
