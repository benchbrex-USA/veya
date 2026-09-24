import sys
from pathlib import Path

from veya.foundry import (
    VeyaBridgeConfig,
    VeyaProcessExpert,
    VeyaQuestion,
    VeyaQuestionType,
    VeyaRequest,
)


def test_process_bridge_round_trip(tmp_path: Path):
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import json, sys\n"
        "for line in sys.stdin:\n"
        "    req=json.loads(line)\n"
        "    q=req['questions'][0]\n"
        "    labels=q['options'] or ['false','true']\n"
        "    p={label: 1.0/len(labels) for label in labels}\n"
        "    out={'expert_id':'source_001','answers':{q['id']:{'probabilities':p}}}\n"
        "    print(json.dumps(out), flush=True)\n",
        encoding="utf-8",
    )
    expert = VeyaProcessExpert(
        VeyaBridgeConfig(expert_id="source_001", command=(sys.executable, str(worker)))
    )
    q = VeyaQuestion(
        question_id="q",
        question_type=VeyaQuestionType.CHOICE,
        instruction="pick",
        options=("x", "y"),
    )
    response = expert.predict(VeyaRequest(state="s", questions=(q,)))
    expert.close()
    assert response.answers["q"].distribution.as_mapping() == {"x": 0.5, "y": 0.5}
