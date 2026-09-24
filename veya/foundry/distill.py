"""Create Veya distillation records from Foundry consensus."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

from .contracts import VeyaRequest
from .pool import VeyaConsensus


@dataclass(frozen=True)
class VeyaDistillationRecord:
    state: Any
    questions: tuple[dict[str, Any], ...]
    targets: Mapping[str, Mapping[str, float]]
    disagreement: Mapping[str, float]
    metadata: Mapping[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "questions": list(self.questions),
            "targets": {k: dict(v) for k, v in self.targets.items()},
            "disagreement": dict(self.disagreement),
            "metadata": dict(self.metadata),
        }


def make_distillation_record(
    request: VeyaRequest,
    consensus: VeyaConsensus,
    *,
    min_experts: int = 2,
) -> VeyaDistillationRecord:
    if consensus.expert_count < min_experts:
        raise ValueError(f"distillation requires at least {min_experts} experts")
    questions = tuple(
        {
            "id": q.question_id,
            "type": q.question_type.value,
            "instruction": q.instruction,
            "options": list(q.options),
            "descriptions": dict(q.option_descriptions),
        }
        for q in request.questions
    )
    targets = {
        question_id: answer.distribution.as_mapping()
        for question_id, answer in consensus.answers.items()
    }
    return VeyaDistillationRecord(
        state=request.state,
        questions=questions,
        targets=targets,
        disagreement=consensus.disagreement,
        metadata={
            "request_id": request.request_id,
            "expert_count": consensus.expert_count,
            **dict(request.metadata),
        },
    )
