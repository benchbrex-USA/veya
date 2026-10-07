"""Presentation perturbations for plot specimens.

Ground truth is copied through unchanged. These transforms describe the plot.
They do not redraw pixels and they do not call a model.
"""

from __future__ import annotations

import copy
from typing import Any

PERTURBATIONS = (
    "resolution",
    "crop",
    "axis_range",
    "color",
    "legend",
    "font",
    "annotation",
    "missing_title",
    "extra_trace",
    "distractor_trace",
    "background_noise",
    "log_scale",
    "unit_representation",
)


def apply_perturbation(specimen: dict[str, Any], name: str) -> dict[str, Any]:
    if name not in PERTURBATIONS:
        raise ValueError("unknown perturbation")
    if "ground_truth" not in specimen:
        raise ValueError("ground truth required")
    updated = copy.deepcopy(specimen)
    if name == "resolution":
        updated["pixel_width"] = 320
        updated["pixel_height"] = 240
    elif name == "crop":
        updated["crop"] = {"x": 0.1, "y": 0.1, "width": 0.8, "height": 0.8}
    elif name == "axis_range":
        axis = dict(updated.get("axis") or {})
        axis["x_max"] = float(axis.get("x_max", 1.0)) * 1.25
        updated["axis"] = axis
    elif name == "color":
        updated["series_color"] = "gray"
    elif name == "legend":
        updated["legend"] = "bottom"
    elif name == "font":
        updated["font"] = "serif"
    elif name == "annotation":
        notes = list(updated.get("annotations") or [])
        notes.append("extra annotation")
        updated["annotations"] = notes
    elif name == "missing_title":
        updated["title"] = ""
    elif name == "extra_trace":
        updated["trace_count"] = int(updated.get("trace_count", 1)) + 1
    elif name == "distractor_trace":
        updated["distractor_trace"] = True
    elif name == "background_noise":
        updated["background_noise"] = True
    elif name == "log_scale":
        updated["y_scale"] = "log"
    elif name == "unit_representation":
        updated["display_unit"] = "MHz"
        updated["display_scale"] = 1000.0
    if updated["ground_truth"] != specimen["ground_truth"]:
        raise RuntimeError("perturbation changed ground truth")
    return updated


def prediction_stability(labels: dict[str, str]) -> dict[str, Any]:
    if "reference" not in labels:
        raise ValueError("reference label required")
    reference = labels["reference"]
    changed = [name for name, label in labels.items() if name != "reference" and label != reference]
    return {"stable": not changed, "changed": changed, "reference": reference}
