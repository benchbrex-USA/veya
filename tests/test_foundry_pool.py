from veya.foundry import (
    VeyaAnswer,
    VeyaDistribution,
    VeyaExpert,
    VeyaExpertPool,
    VeyaQuestion,
    VeyaQuestionType,
    VeyaRequest,
    VeyaResponse,
    make_distillation_record,
)


class FixedExpert(VeyaExpert):
    def __init__(self, expert_id, probabilities):
        self._expert_id = expert_id
        self._probabilities = probabilities

    @property
    def expert_id(self):
        return self._expert_id

    def predict(self, request):
        q = request.questions[0]
        return VeyaResponse(
            expert_id=self.expert_id,
            answers={
                q.question_id: VeyaAnswer(
                    question_id=q.question_id,
                    distribution=VeyaDistribution(
                        labels=q.normalized_options(),
                        probabilities=tuple(self._probabilities),
                    ),
                )
            },
        )


def test_consensus_and_distillation():
    q = VeyaQuestion(
        question_id="route",
        question_type=VeyaQuestionType.CHOICE,
        instruction="choose",
        options=("a", "b"),
    )
    request = VeyaRequest(state="example", questions=(q,), request_id="r1")
    pool = VeyaExpertPool([
        FixedExpert("source_001", (0.8, 0.2)),
        FixedExpert("source_002", (0.6, 0.4)),
    ])
    responses = pool.predict(request)
    consensus = pool.consensus(request, responses)
    assert consensus.answers["route"].distribution.probabilities == (0.7, 0.30000000000000004)
    assert consensus.disagreement["route"] > 0
    record = make_distillation_record(request, consensus)
    assert record.targets["route"]["a"] == 0.7
    assert record.metadata["expert_count"] == 2
