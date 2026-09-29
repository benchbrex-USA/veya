"""Unit tests for the rival head-to-head benchmark. No large model downloads, no network."""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "research"))
sys.path.insert(0, os.path.join(REPO, "research", "scripts"))

from veyabench.rivals.base import (  # noqa: E402
    Rival, first_question, option_labels, option_texts, probs_from_scores,
    render_state, to_rows,
)


CHOICE_Q = {"type": "choice", "instructions": "Pick the topic of `article`.",
            "criteria": {"world": "international news", "sports": None, "business": "economy"}}
NOUL_Q = {"type": "noul", "instructions": "Is `email` spam?",
          "criteria": {"true": "unsolicited bulk", "false": "legitimate"}}
NOUL_Q_BARE = {"type": "noul", "instructions": "Is `x` toxic?"}
SCORE_Q = {"type": "score", "instructions": "How urgent?",
           "criteria": ["can wait", "this week", "today"]}


def test_render_state_dict_and_str():
    assert "article: hello" in render_state({"article": "hello"})
    assert render_state("plain") == "plain"


def test_option_texts_and_labels_align_choice():
    assert option_labels(CHOICE_Q) == ["world", "sports", "business"]
    texts = option_texts(CHOICE_Q)
    assert texts[0] == "world - international news"
    assert texts[1] == "sports"          # None description -> bare label
    assert len(texts) == len(option_labels(CHOICE_Q))


def test_option_labels_noul_default_and_criteria():
    assert option_labels(NOUL_Q_BARE) == ["false", "true"]
    assert option_labels(NOUL_Q) == ["false", "true"]
    assert option_texts(NOUL_Q) == ["legitimate", "unsolicited bulk"]


def test_option_labels_score():
    assert option_labels(SCORE_Q) == ["can wait", "this week", "today"]


def test_probs_from_scores_sums_to_one():
    p = probs_from_scores([1.0, 2.0, 3.0])
    assert abs(p.sum() - 1.0) < 1e-9 and p.shape == (3,)


def test_first_question():
    assert first_question({"topic": CHOICE_Q})["type"] == "choice"
    assert first_question({}) is None


class _StubRival(Rival):
    name = "stub"
    def __init__(self, table):
        self.table = table       # maps frozenset(option_labels) -> probs
    def _options_probs(self, state, qdef, options):
        return self.table[frozenset(option_labels(qdef))]


def test_rival_score_rows_shape_and_metrics_parity():
    from bench_local import metrics
    cases = [({"article": "a goal was scored"}, {"topic": CHOICE_Q}),
             ({"article": "markets fell"}, {"topic": CHOICE_Q})]
    gold = [1, 2]
    stub = _StubRival({frozenset(["world", "sports", "business"]): [0.1, 0.7, 0.2]})
    rows, secs = stub.score(cases, gold)
    assert len(rows) == 2 and secs >= 0.0
    assert all(r[1] is not None and abs(r[1].sum() - 1.0) < 1e-6 for r in rows)
    m = metrics(rows)
    assert 0.0 <= m["accuracy"] <= 1.0 and 0.0 <= m["ece"] <= 1.0


def test_rival_score_drops_bad_cases():
    cases = [({"article": "x"}, {"topic": CHOICE_Q}),
             ({"article": "y"}, {"topic": {"type": "choice", "instructions": "?", "criteria": {}}})]
    stub = _StubRival({frozenset(["world", "sports", "business"]): [0.1, 0.7, 0.2]})
    rows, _ = stub.score(cases, [0, 0])
    assert rows[0][1] is not None and rows[1][1] is None   # empty criteria -> dropped


def test_nli_reindexes_to_option_order():
    from veyabench.rivals.nli import NliRival

    captured = {}
    texts = option_texts(CHOICE_Q)   # ["world - international news", "sports", "business - economy"]

    def fake_pipe(premise, candidate_labels=None, hypothesis_template=None, multi_label=None):
        captured["labels"] = list(candidate_labels)
        # The real pipeline echoes candidate_labels back SORTED BY SCORE (desc); the adapter
        # must undo that ordering. Fake returns business(0.6) > world(0.3) > sports(0.1).
        order = [candidate_labels[2], candidate_labels[0], candidate_labels[1]]
        return {"labels": order, "scores": [0.6, 0.3, 0.1]}

    r = NliRival()
    r._pipe = fake_pipe                      # inject stub, skip the 1.6 GB download
    probs = r._options_probs({"article": "x"}, CHOICE_Q, texts)
    labels = option_labels(CHOICE_Q)         # ["world", "sports", "business"]
    assert probs[labels.index("world")] == pytest.approx(0.3)
    assert probs[labels.index("sports")] == pytest.approx(0.1)
    assert probs[labels.index("business")] == pytest.approx(0.6)
    assert captured["labels"] == texts


def test_nli_availability_is_bool():
    from veyabench.rivals.nli import NliRival
    assert isinstance(NliRival().available(), bool)


def test_st_maps_similarity_to_distribution():
    from veyabench.rivals.st import StCentroidRival

    class FakeST:
        def encode(self, texts, convert_to_numpy=False, normalize_embeddings=False):
            # premise + 3 options; make option index 2 most similar to premise
            import numpy as _np
            n = len(texts)
            m = _np.zeros((n, 4), dtype=float)
            m[0] = [1.0, 0.0, 0.0, 0.0]              # premise
            m[1] = [0.2, 1.0, 0.0, 0.0]              # option 0 (low sim)
            m[2] = [0.3, 0.0, 1.0, 0.0]              # option 1 (low sim)
            m[3] = [0.95, 0.05, 0.0, 0.0]            # option 2 (high sim)
            return m / _np.linalg.norm(m, axis=1, keepdims=True)

    r = StCentroidRival()
    r._model = FakeST()
    probs = r._options_probs({"article": "x"}, CHOICE_Q, option_texts(CHOICE_Q))
    import numpy as _np
    p = _np.asarray(probs, float)
    assert p.shape == (3,) and int(p.argmax()) == 2   # highest similarity wins


@pytest.mark.skipif(os.environ.get("RUN_MODEL_DOWNLOADS") != "1",
                    reason="set RUN_MODEL_DOWNLOADS=1 to fetch all-MiniLM-L6-v2")
def test_st_smoke_real_model():
    from veyabench.rivals.st import StCentroidRival
    r = StCentroidRival(device="cpu")
    rows, secs = r.score([({"article": "a goal was scored"}, {"topic": CHOICE_Q})], [1])
    assert rows[0][1] is not None and abs(rows[0][1].sum() - 1.0) < 1e-6


def test_jev_dormant_without_credential(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    from veyabench.rivals.jev import JevRival
    r = JevRival()
    assert r.available() is False
    assert "TYPESAFE_API_KEY" in r.unavailable_reason()


def test_jev_caps_cases(monkeypatch):
    monkeypatch.setenv("JEV_MAX_CASES", "1")
    from veyabench.rivals.jev import JevRival
    r = JevRival()
    assert r.max_cases == 1


def test_require_datasets_exits_when_missing(monkeypatch):
    import importlib.util as _u
    import bench_rival
    real = _u.find_spec

    def fake(name, *a, **k):
        return None if name == "datasets" else real(name, *a, **k)

    monkeypatch.setattr(bench_rival.importlib.util, "find_spec", fake)
    with pytest.raises(SystemExit) as ei:
        bench_rival._require_datasets()
    assert ei.value.code == 2


def test_select_rivals_skips_unavailable():
    import bench_rival
    rivals = bench_rival.select_rivals("all", device=None)
    names = {r.name for r in rivals}
    assert "jev-typesafe" not in names          # no credential on this machine
    assert names <= {"nli-bart", "st-centroid"}
