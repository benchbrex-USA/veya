"""Primary verdict orchestrator for the blind shadow evaluation protocol.

``run_blind_verdict(config)`` is the run-once entry point of the protocol: it
composes the freeze record, the contamination screens, the per-slice
selective gates, the per-slice calibration reports, the sample-size check,
and the caller's operational measurements into one verdict document that
carries the Section-16 result language.

    P1 data validity          contamination screens over blind vs controlled training
    P2 selective safety       clean-slice risk UCB95 <= frozen epsilon (G1)
    P3 useful coverage        clean-slice coverage >= frozen c_min
    P4 shift robustness       G2 + G3 + G4 over the pre-registered shift slices
    P5 calibration transfer   frozen temperature artifact; never refit on blind data
    P6 operational viability  caller-supplied operational measurements

Nothing here selects a threshold, fits a temperature, or tunes anything on
blind data: the frozen selective policy is applied unchanged, ``epsilon`` and
``c_min`` are read from the validated freeze record, and a config that
disagrees with the record is rejected before any statistic is computed.

Split provenance: a config that declares ``calibration_data``/``test_data``
pins (mappings with ``path`` and ``sha256``) must declare both, and the two
datasets must differ in path and in content hash. The check runs before the
contamination screens and every statistic, and the verdict echoes the
validated pins as ``split_provenance``.

Case contract: every case is a mapping with ``text``, ``gold_label``,
``prediction``, ``correct``, ``probabilities``, ``logits`` (aligned with
``probabilities``), and ``slice_tag``, plus optional ``confidence`` (defaults
to the max probability) and ``candidate_count`` (checked against the
probability width when present). Integer ``gold_label``/``prediction`` values
are used directly as logit column indices; string labels are mapped through
the sorted vocabulary of all gold and predicted labels in the evaluation.
"""
from __future__ import annotations

import re
from typing import Any, Mapping

import numpy as np

from research.veyabench.calibration_report import calibration_report, calibration_summary
from research.veyabench.contamination import contamination_report
from research.veyabench.freeze_record import freeze_record_hash, validate_freeze_record
from research.veyabench.paraphrase_noise import CONDITIONS as S5_CONDITIONS
from research.veyabench.paraphrase_noise import perturb as s5_perturb
from research.veyabench.phase11_noise import CONDITIONS as P11_CONDITIONS
from research.veyabench.phase11_noise import perturb as p11_perturb
from research.veyabench.phase6_protocol import paired_bootstrap
from research.veyabench.sample_size import required_n
from research.veyabench.selective_gates import CLEAN_SLICE, gate_report
from research.veyabench.surface_noise import CONDITIONS as S4_CONDITIONS
from research.veyabench.surface_noise import perturb as s4_perturb
from veya.selective import score_probabilities

BLIND_VERDICT_FORMAT = "veya-blind-verdict-v1"

PRIMARY_GATE_NAMES = (
    "P1_data_validity",
    "P2_selective_safety",
    "P3_useful_coverage",
    "P4_shift_robustness",
    "P5_calibration_transfer",
    "P6_operational_viability",
)

DEFAULT_COVERAGE_RETENTION_MIN = 0.95
DEFAULT_COLLAPSE_FLOOR = {"coverage": 0.05, "accuracy": 0.30}
DEFAULT_CONTAMINATION_TOLERANCE = 0

# Which registered noise family each protocol slice condition comes from.
SLICE_FAMILIES = {
    **{condition: "phase11-generic-noise-v1" for condition in P11_CONDITIONS},
    **{condition: "blind-surface-noise-v1" for condition in S4_CONDITIONS},
    **{condition: "blind-template-paraphrase-v1" for condition in S5_CONDITIONS},
    CLEAN_SLICE: "clean",
}
KNOWN_SLICE_CONDITIONS = ("clean",) + tuple(
    condition
    for condition in (*P11_CONDITIONS, *S4_CONDITIONS, *S5_CONDITIONS)
    if condition != CLEAN_SLICE
)

_REQUIRED_CASE_FIELDS = (
    "text",
    "gold_label",
    "prediction",
    "correct",
    "probabilities",
    "logits",
    "slice_tag",
)
_OPERATIONAL_FAILURE_KEYS = ("failures", "failed_checks", "violations")
_HEX64_RE = re.compile(r"\A[0-9a-f]{64}\Z")


def perturb_text(text: str, condition: str, row_id: str) -> str:
    """Apply the registered noise perturbation for ``condition`` to ``text``.

    Phase-11 generic noise takes precedence, then the S4 surface and S5
    paraphrase conditions. ``clean`` returns the stripped source text under
    every family, so its position in the dispatch order is immaterial.
    """
    if condition in P11_CONDITIONS:
        return p11_perturb(text, condition, row_id)
    if condition in S4_CONDITIONS:
        return s4_perturb(text, condition, row_id)
    if condition in S5_CONDITIONS:
        return s5_perturb(text, condition, row_id)
    raise ValueError("unknown slice condition %r" % (condition,))


def _percent(value) -> str:
    return "n/a" if value is None else "%.1f" % (float(value) * 100.0)


def _all_int_labels(values) -> bool:
    return all(
        isinstance(value, (int, np.integer)) and not isinstance(value, bool)
        for value in values
    )


def _case_confidence(case: Mapping[str, Any]) -> float:
    value = case.get("confidence")
    if value is not None:
        return float(value)
    probabilities = np.asarray(case["probabilities"], dtype=np.float64)
    return float(probabilities.max())


def _validate_cases(cases) -> None:
    if not cases:
        raise ValueError("the blind verdict needs at least one evaluated case")
    for position, case in enumerate(cases):
        if not isinstance(case, Mapping):
            raise ValueError("case %d is not a mapping" % position)
        missing = [field for field in _REQUIRED_CASE_FIELDS if field not in case]
        if missing:
            raise ValueError(
                "case %d is missing fields: %s" % (position, ", ".join(missing))
            )
        candidate_count = case.get("candidate_count")
        if candidate_count is not None and int(candidate_count) != len(
            case["probabilities"]
        ):
            raise ValueError(
                "case %d candidate_count %r does not match its probability width %d"
                % (position, candidate_count, len(case["probabilities"]))
            )


def _validate_split_provenance(config: Mapping[str, Any]) -> dict[str, Any] | None:
    """Fail closed unless declared calibration and test datasets are disjoint.

    The prospective protocol evaluates a supplied test dataset against a frozen
    calibration dataset; both sides are declared in the config as mappings with
    ``path`` and ``sha256``. Declaring only one side, reusing one path for both
    sides, or reusing one content hash for both sides is rejected before any
    screen or statistic runs. A config that declares neither side (legacy
    callers) returns ``None`` and the verdict reports no split provenance.
    """
    calibration = config.get("calibration_data")
    test = config.get("test_data")
    if calibration is None and test is None:
        return None
    if calibration is None or test is None:
        missing = "test_data" if calibration is not None else "calibration_data"
        raise ValueError(
            "split provenance requires both calibration_data and test_data; "
            "%s is missing" % missing
        )
    pins: dict[str, dict[str, str]] = {}
    for name, entry in (("calibration_data", calibration), ("test_data", test)):
        if not isinstance(entry, Mapping):
            raise ValueError("%s must be a mapping with path and sha256" % name)
        path = entry.get("path")
        digest = entry.get("sha256")
        if not isinstance(path, str) or not path.strip():
            raise ValueError("%s.path must be a non-empty string" % name)
        if not isinstance(digest, str) or not _HEX64_RE.match(digest):
            raise ValueError(
                "%s.sha256 must be a 64-character lowercase hex SHA-256" % name
            )
        pins[name] = {"path": path, "sha256": digest}
    if pins["calibration_data"]["path"] == pins["test_data"]["path"]:
        raise ValueError(
            "calibration_data and test_data must be distinct datasets; "
            "both declare path %r" % pins["calibration_data"]["path"]
        )
    if pins["calibration_data"]["sha256"] == pins["test_data"]["sha256"]:
        raise ValueError(
            "calibration_data and test_data must be distinct datasets; "
            "both pin content hash %s" % pins["calibration_data"]["sha256"]
        )
    return {
        "calibration_data": pins["calibration_data"],
        "test_data": pins["test_data"],
        "disjoint": True,
    }


def _gold_columns(cases) -> list:
    """Logit column index of every case's gold label.

    Integer gold and prediction labels are used directly as column indices;
    string labels are mapped through one global sorted vocabulary so the
    mapping is identical in every slice.
    """
    golds = [case["gold_label"] for case in cases]
    preds = [case["prediction"] for case in cases]
    if _all_int_labels(golds) and _all_int_labels(preds):
        return [int(gold) for gold in golds]
    vocabulary = {
        label: index
        for index, label in enumerate(
            sorted({str(gold) for gold in golds} | {str(pred) for pred in preds})
        )
    }
    return [vocabulary[str(gold)] for gold in golds]


def _calibration_payloads(cases, policy, temperature) -> dict:
    """Group cases by slice and build ``calibration_report`` payloads.

    ``scores`` reuses the frozen policy's selective score so the calibration
    report's risk-coverage curve and accepted/deferred split describe the
    same operating point the gates evaluated.
    """
    score_kind = policy["score"]
    columns = _gold_columns(cases)
    groups: dict[str, list] = {}
    for case, column in zip(cases, columns):
        groups.setdefault(str(case["slice_tag"]), []).append((case, column))
    payloads = {}
    for tag, pairs in groups.items():
        group = [case for case, _column in pairs]
        payloads[tag] = {
            "logits": np.asarray([case["logits"] for case in group], dtype=np.float64),
            "labels": np.asarray(
                [column for _case, column in pairs], dtype=np.int64
            ),
            "confidence": np.asarray(
                [_case_confidence(case) for case in group], dtype=np.float64
            ),
            "correct": np.asarray(
                [1.0 if case["correct"] else 0.0 for case in group],
                dtype=np.float64,
            ),
            "scores": np.asarray(
                [score_probabilities(case["probabilities"], score_kind) for case in group],
                dtype=np.float64,
            ),
            "temperature": float(temperature),
        }
    return payloads


def _operating_threshold(policy) -> float | None:
    if not isinstance(policy, Mapping):
        raise ValueError("selective policy must be a mapping")
    operating = policy.get("operating_epsilon")
    thresholds = policy.get("thresholds")
    if not isinstance(thresholds, Mapping) or operating not in thresholds:
        raise ValueError(
            "selective policy is missing its operating threshold for %r" % (operating,)
        )
    value = thresholds[operating]
    return None if value is None else float(value)


def _per_class_metrics(cases) -> tuple:
    """Per-class support/accuracy plus precision/recall/F1 from label equality.

    ``accuracy`` counts the case's own ``correct`` flag inside its gold class;
    precision, recall, and F1 compare ``gold_label`` with ``prediction``, so
    the two views agree exactly when the caller's data is self-consistent.
    """
    golds = [str(case["gold_label"]) for case in cases]
    preds = [str(case["prediction"]) for case in cases]
    classes = sorted(set(golds) | set(preds))
    per_class = {}
    f1_values = []
    for label in classes:
        tp = fp = fn = support = correct_count = 0
        for gold, pred, case in zip(golds, preds, cases):
            is_gold = gold == label
            is_pred = pred == label
            if is_gold:
                support += 1
                if case["correct"]:
                    correct_count += 1
                if is_pred:
                    tp += 1
                else:
                    fn += 1
            elif is_pred:
                fp += 1
        denominator = 2 * tp + fp + fn
        f1 = (2.0 * tp / denominator) if denominator else 0.0
        per_class[label] = {
            "support": support,
            "correct": correct_count,
            "accuracy": (correct_count / support) if support else None,
            "precision": (tp / (tp + fp)) if (tp + fp) else None,
            "recall": (tp / (tp + fn)) if (tp + fn) else None,
            "f1": float(f1),
        }
        if support:
            f1_values.append(f1)
    macro_f1 = (sum(f1_values) / len(f1_values)) if f1_values else None
    return per_class, macro_f1


def _secondary_metrics(clean_cases, clean_calibration, record) -> dict:
    """Clean-slice accuracy, macro F1, ECE, Brier, per-class rows, bootstrap CI.

    The bootstrap compares the observed correctness vector against an
    all-correct baseline: the paired delta mean is ``accuracy - 1``, so the
    delta interval shifted by one is the accuracy interval.
    """
    n = len(clean_cases)
    if not n:
        return {
            "accuracy": None,
            "macro_f1": None,
            "ece": None,
            "brier": None,
            "per_class": {},
            "bootstrap_ci": None,
        }
    correct = np.asarray(
        [1.0 if case["correct"] else 0.0 for case in clean_cases], dtype=np.float64
    )
    per_class, macro_f1 = _per_class_metrics(clean_cases)
    ece = clean_calibration.get("ece") if clean_calibration else None
    brier = clean_calibration.get("brier") if clean_calibration else None
    draws = int(record["bootstrap_draws"])
    seed = int(record["random_seeds"].get("bootstrap", 17))
    bootstrap = paired_bootstrap(np.ones(n, dtype=np.float64), correct, n=draws, seed=seed)
    bootstrap_ci = {
        "baseline": "all-correct",
        "delta": float(bootstrap["delta"]),
        "ci95_low": float(bootstrap["ci95_low"]),
        "ci95_high": float(bootstrap["ci95_high"]),
        "n": int(bootstrap["n"]),
        "draws": draws,
        "seed": seed,
        "accuracy": 1.0 + float(bootstrap["delta"]),
        "accuracy_ci95_low": max(0.0, 1.0 + float(bootstrap["ci95_low"])),
        "accuracy_ci95_high": min(1.0, 1.0 + float(bootstrap["ci95_high"])),
    }
    return {
        "accuracy": float(correct.mean()),
        "macro_f1": macro_f1,
        "ece": None if ece is None else float(ece),
        "brier": None if brier is None else float(brier),
        "per_class": per_class,
        "bootstrap_ci": bootstrap_ci,
    }


def _operational_failures(operational) -> list:
    if operational is None:
        return ["no operational measurements were provided"]
    if not isinstance(operational, Mapping):
        raise ValueError("operational measurements must be a mapping")
    failures = []
    if operational.get("passed") is False:
        failures.append("operational.passed is False")
    if operational.get("failed") is True:
        failures.append("operational.failed is True")
    for key in _OPERATIONAL_FAILURE_KEYS:
        value = operational.get(key)
        if isinstance(value, (list, tuple)) and value:
            failures.append("operational.%s reports %d item(s)" % (key, len(value)))
        elif value:
            failures.append("operational.%s is set" % key)
    return failures


def _shift_failure_slices(gates_report) -> list:
    """Sorted union of slice tags that broke any shift gate (G2/G3/G4)."""
    gates = gates_report.get("gates") or {}
    tags = set()
    for name in ("G2_shift_safety", "G3_coverage_retention"):
        gate = gates.get(name) or {}
        tags.update(str(tag) for tag in gate.get("failing_slices") or [])
    for violation in (gates.get("G4_collapse_floor") or {}).get("violations") or []:
        tags.add(str(violation.get("slice_tag")))
    return sorted(tags)


def _result_language(primary, gates_report, clean_stats, domain) -> str:
    """Section-16 result language, with failure precedence P1 > P2 > P3 > P4 > P5 > P6."""
    if all(primary.values()):
        return (
            "On prospectively collected %s traffic, using the preregistered frozen "
            "configuration, Veya automatically handled %s%% of eligible English "
            "≤19-candidate Choice decisions while maintaining an observed error rate "
            "of %s%%, with a 95%% upper confidence bound of %s%%. The predefined "
            "production-shift slices also remained within the registered safety and "
            "coverage-retention bounds. Remaining cases were deferred."
            % (
                domain,
                _percent(clean_stats.get("coverage")),
                _percent(clean_stats.get("observed_risk")),
                _percent(clean_stats.get("risk_ucb95")),
            )
        )
    if not primary["P1_data_validity"]:
        return (
            "The blind evaluation data failed the preregistered contamination "
            "screens, so no selective-automation claim is supportable on this traffic."
        )
    if not primary["P2_selective_safety"]:
        return (
            "The frozen selective policy did not maintain the preregistered risk "
            "bound on prospective traffic."
        )
    if not primary["P3_useful_coverage"]:
        return (
            "Veya met the risk target but did not automate enough traffic to meet "
            "the preregistered usefulness threshold."
        )
    if not primary["P4_shift_robustness"]:
        slices = ", ".join(_shift_failure_slices(gates_report)) or "the evaluated shift slices"
        return (
            "Clean-domain selective performance transferred, but the system did not "
            "satisfy the preregistered robustness requirements under %s." % slices
        )
    if not primary["P5_calibration_transfer"]:
        return (
            "The frozen calibration artifact could not be verified for this evaluation."
        )
    return "Operational viability requirements were not met on prospective traffic."


def _sample_size_check(clean_stats, epsilon, c_min, record) -> dict:
    """Section-7 check: enough accepted clean cases for the epsilon bound.

    ``required_n`` asks how many accepted cases the observed clean error count
    needs for its Wilson upper bound to stay at or below epsilon, assuming the
    preregistered minimum coverage ``c_min``.
    """
    actual_accepted = int(clean_stats.get("accepted") or 0)
    max_errors = int(clean_stats.get("incorrect") or 0)
    confidence = float(record["bootstrap_confidence"])
    requirement = required_n(epsilon, c_min, max_errors, confidence=confidence)
    required_accepted = int(requirement["accepted"])
    return {
        "epsilon": float(epsilon),
        "confidence": confidence,
        "max_errors": max_errors,
        "anticipated_coverage": float(c_min),
        "required_accepted": required_accepted,
        "required_total": int(requirement["total"]),
        "actual_accepted": actual_accepted,
        "sufficient": bool(actual_accepted >= required_accepted),
        "wilson_ucb_at_accepted": float(requirement["wilson_ucb_at_accepted"]),
    }


def run_blind_verdict(config: Mapping[str, Any]) -> dict[str, Any]:
    """Run the full blind protocol once and return the structured verdict.

    ``config`` carries the pre-built freeze record, the evaluated cases, the
    training-side texts (and optional embeddings) for the contamination
    screens, the frozen selective policy, the protocol constants, and optional
    operational measurements. Optional ``calibration_data``/``test_data``
    entries (mappings with ``path`` and ``sha256``) pin the two datasets;
    declaring them requires both sides to be present and disjoint, and the
    verdict echoes the pins as ``split_provenance``. The result matches
    ``veya-blind-verdict-v1``: see the module docstring for the case contract
    and the verdict shape.
    """
    if not isinstance(config, Mapping):
        raise ValueError("config must be a mapping")
    for key in ("freeze_record", "cases", "training_texts", "policy"):
        if key not in config:
            raise ValueError("config is missing required key %r" % key)

    # P0: the freeze record is valid, hashed, and agrees with the config.
    record = config["freeze_record"]
    validate_freeze_record(record)
    record_hash = freeze_record_hash(record)
    epsilon = float(record["epsilon"])
    c_min = float(record["c_min"])
    for key, frozen in (("epsilon", epsilon), ("c_min", c_min)):
        if key in config and float(config[key]) != frozen:
            raise ValueError(
                "config %s %r does not match the frozen record value %r"
                % (key, config[key], frozen)
            )
    retention_min = float(config.get("coverage_retention_min", DEFAULT_COVERAGE_RETENTION_MIN))
    collapse_floor = dict(config.get("collapse_floor") or DEFAULT_COLLAPSE_FLOOR)
    tolerance = int(config.get("contamination_tolerance", DEFAULT_CONTAMINATION_TOLERANCE))
    temperature = float(config.get("temperature", 1.0))
    domain = str(config.get("domain", "unspecified"))
    policy = config["policy"]
    cases = list(config["cases"])
    _validate_cases(cases)

    # P0.5: declared calibration/test datasets must be present together and
    # disjoint. This provenance check runs before the contamination screens
    # and every statistic; the verdict echoes the validated pins.
    split_provenance = _validate_split_provenance(config)

    # P1: contamination screens over the blind texts vs the controlled training side.
    blind_texts = [str(case["text"]) for case in cases]
    training_texts = [str(text) for text in (config["training_texts"] or [])]
    contamination = contamination_report(
        blind_texts,
        training_texts,
        config.get("blind_vectors"),
        config.get("training_vectors"),
    )

    # P2 + P4: per-slice selective statistics and the G1-G4 gates.
    gates = gate_report(cases, policy, epsilon, retention_min, collapse_floor)
    clean_stats = gates["slices"].get(CLEAN_SLICE) or {}
    clean_cases = [case for case in cases if str(case["slice_tag"]) == CLEAN_SLICE]

    # P3: the clean slice must automate enough traffic to be useful.
    useful_coverage = bool(
        clean_stats.get("coverage") is not None
        and float(clean_stats["coverage"]) >= c_min
    )

    # P5 evidence: per-slice calibration under the frozen temperature.
    threshold = _operating_threshold(policy)
    payloads = _calibration_payloads(cases, policy, temperature)
    calibration = calibration_summary(calibration_report(payloads, threshold=threshold))

    # P6: operational viability is the caller's measurement to supply.
    operational = config.get("operational")
    operational_failures = _operational_failures(operational)

    gate_results = gates["gates"]
    primary_gates = {
        "P1_data_validity": bool(len(contamination["total_flagged"]) <= tolerance),
        "P2_selective_safety": bool(gate_results["G1_clean_safety"]["passed"]),
        "P3_useful_coverage": useful_coverage,
        "P4_shift_robustness": bool(
            all(
                gate_results[name]["passed"]
                for name in ("G2_shift_safety", "G3_coverage_retention", "G4_collapse_floor")
            )
        ),
        # By construction the temperature was frozen before the blind run; the
        # gate only verifies the record pins the calibration artifact.
        "P5_calibration_transfer": bool(record.get("temperature_artifact_sha256")),
        "P6_operational_viability": bool(operational is not None and not operational_failures),
    }
    all_primary_gates_pass = bool(all(primary_gates.values()))

    secondary = _secondary_metrics(
        clean_cases, (calibration.get("slices") or {}).get(CLEAN_SLICE), record
    )
    sample_size_check = _sample_size_check(clean_stats, epsilon, c_min, record)
    result_language = _result_language(primary_gates, gates, clean_stats, domain)

    return {
        "format": BLIND_VERDICT_FORMAT,
        "freeze_record_hash": record_hash,
        "split_provenance": split_provenance,
        "contamination": contamination,
        "gates": gates,
        "calibration": calibration,
        "primary_gates": primary_gates,
        "all_primary_gates_pass": all_primary_gates_pass,
        "secondary": secondary,
        "result_language": result_language,
        "sample_size_check": sample_size_check,
        "operational": {
            "present": operational is not None,
            "failures": operational_failures,
            "measurements": dict(operational) if isinstance(operational, Mapping) else None,
        },
        "slice_families": {
            tag: SLICE_FAMILIES.get(tag, "unregistered")
            for tag in gates["slices"]
        },
    }


def verdict_summary(verdict: Mapping[str, Any]) -> str:
    """One-line human-readable pass/fail summary of a verdict document."""
    primary = verdict.get("primary_gates") or {}
    failed = [name for name, passed in primary.items() if not passed]
    stamp = str(verdict.get("freeze_record_hash") or "")[:12] or "unknown"
    if verdict.get("all_primary_gates_pass"):
        clean = ((verdict.get("gates") or {}).get("slices") or {}).get(CLEAN_SLICE) or {}
        return (
            "PASS: all six primary gates hold; clean coverage %s%%, observed risk %s%%, "
            "UCB95 %s%% (freeze %s)"
            % (
                _percent(clean.get("coverage")),
                _percent(clean.get("observed_risk")),
                _percent(clean.get("risk_ucb95")),
                stamp,
            )
        )
    return "FAIL: %s (freeze %s)" % (
        ", ".join(failed) if failed else "no primary gates recorded",
        stamp,
    )
