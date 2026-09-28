"""Schema-conditioned scorer must respect the three causal context modes."""
from research.veyabench.run_phase12_sparse_scorer import context_fields


def test_feature_context_modes_preserve_speaker_and_scope():
    row = {
        "current": "yes",
        "history": (("USER", "find flights"), ("SYSTEM", "shall I book it")),
    }
    current = context_fields(row, "utterance_only")
    previous = context_fields(row, "previous_turn")
    full = context_fields(row, "full_context")
    assert [name for name, _ in current] == ["current"]
    assert [name for name, _ in previous] == ["current", "previous_system"]
    assert "flights" not in set().union(*(words for _, words in previous))
    assert any(name == "all_user" and "flights" in words for name, words in full)
    assert any(name == "recent_system" and "book" in words for name, words in full)
