"""Multi-expert execution and disagreement mining for Veya."""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence

from .contracts import VeyaAnswer, VeyaDistribution, VeyaRequest, VeyaResponse
from .expert import VeyaExpert


@dataclass(frozen=True)
class VeyaConsensus:
    answers: Mapping[str, VeyaAnswer]
    disagreement: Mapping[str, float]
    expert_count: int


def _aligned_distribution(answer: VeyaAnswer, labels: Sequence[str]) -> list[float]:
    raw = answer.distribution.as_mapping()
    return [float(raw.get(label, 0.0)) for label in labels]


def _js_divergence(distributions: Sequence[Sequence[float]]) -> float:
    if len(distributions) <= 1:
        return 0.0
    width = len(distributions[0])
    mean = [sum(row[i] for row in distributions) / len(distributions) for i in range(width)]
    eps = 1e-12

    def kl(p: Sequence[float], q: Sequence[float]) -> float:
        return sum(pi * math.log(max(pi, eps) / max(qi, eps)) for pi, qi in zip(p, q) if pi > 0)

    return sum(kl(row, mean) for row in distributions) / len(distributions)


class VeyaExpertPool:
    def __init__(self, experts: Iterable[VeyaExpert]):
        experts = tuple(experts)
        ids = [expert.expert_id for expert in experts]
        if not experts:
            raise ValueError("VeyaExpertPool requires at least one expert")
        if len(ids) != len(set(ids)):
            raise ValueError("Veya expert ids must be unique")
        self.experts = experts

    def predict(self, request: VeyaRequest, max_workers: int | None = None) -> tuple[VeyaResponse, ...]:
        workers = max_workers or len(self.experts)
        results: Dict[str, VeyaResponse] = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(expert.predict, request): expert.expert_id for expert in self.experts}
            for future in as_completed(futures):
                expert_id = futures[future]
                results[expert_id] = future.result()
        return tuple(results[expert.expert_id] for expert in self.experts)

    @staticmethod
    def consensus(request: VeyaRequest, responses: Sequence[VeyaResponse]) -> VeyaConsensus:
        if not responses:
            raise ValueError("at least one response is required")
        answer_map: Dict[str, VeyaAnswer] = {}
        disagreement: Dict[str, float] = {}
        for question in request.questions:
            labels = question.normalized_options()
            rows = []
            for response in responses:
                if question.question_id not in response.answers:
                    raise ValueError(
                        f"expert {response.expert_id!r} omitted question {question.question_id!r}"
                    )
                rows.append(_aligned_distribution(response.answers[question.question_id], labels))
            mean = [sum(row[i] for row in rows) / len(rows) for i in range(len(labels))]
            total = sum(mean)
            if total <= 0:
                raise ValueError(f"experts produced no mass for {question.question_id!r}")
            mean = [value / total for value in mean]
            answer_map[question.question_id] = VeyaAnswer(
                question_id=question.question_id,
                distribution=VeyaDistribution(labels=tuple(labels), probabilities=tuple(mean)),
                expected_score=(
                    sum(i * p for i, p in enumerate(mean))
                    if question.question_type.value == "score"
                    else None
                ),
            )
            disagreement[question.question_id] = _js_divergence(rows)
        return VeyaConsensus(
            answers=answer_map,
            disagreement=disagreement,
            expert_count=len(responses),
        )

    def close(self) -> None:
        for expert in self.experts:
            expert.close()
