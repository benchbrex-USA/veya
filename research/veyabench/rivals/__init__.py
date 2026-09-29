"""Rival adapters for the measured head-to-head benchmark. Research-only."""
from .base import (Rival, first_question, option_labels, option_texts,
                   probs_from_scores, render_state, to_rows)

REGISTRY = {}   # populated by nli/st/jev modules on import in the driver

__all__ = ["Rival", "REGISTRY", "first_question", "option_labels", "option_texts",
           "probs_from_scores", "render_state", "to_rows"]
