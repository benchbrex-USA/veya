"""Rival adapters for the measured head-to-head benchmark. Research-only."""
from .base import (Rival, first_question, option_labels, option_texts,
                   probs_from_scores, render_state, to_rows)

from .nli import NliRival

REGISTRY = {"nli": NliRival}

__all__ = ["Rival", "REGISTRY", "NliRival", "first_question", "option_labels",
           "option_texts", "probs_from_scores", "render_state", "to_rows"]
