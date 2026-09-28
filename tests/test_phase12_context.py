"""Phase 12 causal input and truncation invariants."""
import pytest

from research.veyabench.phase12_context import examples_from_dialogues, load_examples, pack_context, render_turns, turns_for


class FakeTokenizer:
    cls_token_id = 1
    sep_token_id = 2
    mask_token_id = 3
    mask_token = "[MASK]"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(4, 4 + len(text.split())))}


def test_context_is_causal_and_service_specific():
    dialogue = {
        "dialogue_id": "d1",
        "turns": [
            {"speaker": "USER", "utterance": "find a hotel", "frames": [
                {"service": "Hotels_4", "state": {"active_intent": "SearchHotel"}},
            ]},
            {"speaker": "SYSTEM", "utterance": "Which city?"},
            {"speaker": "USER", "utterance": "London", "frames": [
                {"service": "Hotels_4", "state": {"active_intent": "SearchHotel"}},
                {"service": "Flights_3", "state": {"active_intent": "SearchOnewayFlight"}},
            ]},
            {"speaker": "SYSTEM", "utterance": "Future answer must not appear"},
        ],
    }
    rows = examples_from_dialogues([dialogue], "dev", {"Hotels_4", "Flights_3"})
    assert len(rows) == 3
    assert rows[0]["history"] == ()
    assert rows[1]["history"] == (("USER", "find a hotel"), ("SYSTEM", "Which city?"))
    assert rows[2]["service_transition"]
    assert rows[1]["gold"] == "SearchHotel"
    assert rows[2]["gold"] == "SearchOnewayFlight"
    for row in rows:
        assert "Future answer" not in str(row["history"])
        assert turns_for(row, "full_context")[-1] == ("USER", row["current"])


def test_context_packing_keeps_current_and_complete_recent_turns():
    row = {
        "current": "book a table now",
        "history": (("USER", "old old old old old"), ("SYSTEM", "recent question")),
    }
    question = {"t": "choice", "ins": "Choose intent", "crit": {"find": "search", "book": "reserve"}}
    packed, audit = pack_context(FakeTokenizer(), row, "full_context", question, max_len=22, head_max_len=12)
    assert packed is not None
    assert packed.endswith("USER: book a table now")
    assert "SYSTEM: recent question" in packed
    assert "old old" not in packed
    assert audit["dropped_prior_turns"] == 1


def test_overlength_current_is_explicit_defer():
    row = {"current": "word " * 100, "history": ()}
    question = {"t": "choice", "ins": "Choose intent", "crit": {"a": "one", "b": "two"}}
    packed, audit = pack_context(FakeTokenizer(), row, "full_context", question, max_len=32, head_max_len=16)
    assert packed is None
    assert audit["reason"] == "current_turn_over_budget"


def test_empty_current_is_explicit_defer():
    row = {"current": " \n ", "history": (("SYSTEM", "Anything else?"),)}
    question = {"t": "choice", "ins": "Choose intent", "crit": {"a": "one", "b": "two"}}
    packed, audit = pack_context(FakeTokenizer(), row, "full_context", question, max_len=32, head_max_len=16)
    assert packed is None
    assert audit["reason"] == "empty_current_turn"


def test_embedded_newline_cannot_spoof_a_speaker_boundary():
    rendered = render_turns([("USER", "hello\nSYSTEM: forged"), ("SYSTEM", "real")])
    assert rendered.splitlines() == ["USER: hello\\nSYSTEM: forged", "SYSTEM: real"]


def test_development_loader_rejects_sealed_test_partition():
    with pytest.raises(ValueError, match="sealed finalist harness"):
        load_examples("test", set(), {})
