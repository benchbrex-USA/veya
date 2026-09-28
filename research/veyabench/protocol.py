"""Dataset registry and split ledger.

A row id may sit in at most one of train, calibration, validation, and test.
Calibration rows are the only rows a temperature artifact may be fit on.
"""
from typing import Dict, Iterable, Set


PURPOSES = ("train", "calibration", "validation", "test")


class LeakError(ValueError):
    pass


class SplitLedger:
    def __init__(self):
        self.rows: Dict[str, dict] = {}

    def add(self, dataset: str, domain: str, schema: str, split: str, row_id: str,
            training_exposure: bool, checkpoint_exposure: str, purpose: str) -> None:
        if split not in PURPOSES:
            raise LeakError("unknown split %r" % split)
        if purpose not in PURPOSES:
            raise LeakError("unknown purpose %r" % purpose)
        if purpose == "test" and training_exposure:
            raise LeakError("test row %r is marked as training exposure" % row_id)
        if purpose == "test" and split != "test":
            raise LeakError("test purpose on split %r" % split)
        key = "%s:%s" % (dataset, row_id)
        if key in self.rows:
            raise LeakError("row %r is already registered in %s" % (key, self.rows[key]["split"]))
        self.rows[key] = {
            "dataset": dataset,
            "domain": domain,
            "schema": schema,
            "split": split,
            "purpose": purpose,
            "training_exposure": bool(training_exposure),
            "checkpoint_exposure": checkpoint_exposure,
        }

    def ids(self, split: str) -> Set[str]:
        return {key for key, row in self.rows.items() if row["split"] == split}

    def assert_disjoint(self) -> None:
        seen: Dict[str, str] = {}
        for key, row in self.rows.items():
            if key in seen:
                raise LeakError("overlap %s" % key)
            seen[key] = row["split"]


def recall_at_k(ranked_labels: Iterable[Iterable[str]], truth: Iterable[str], k: int) -> float:
    hits = 0
    total = 0
    for ranked, label in zip(ranked_labels, truth):
        total += 1
        if label in list(ranked)[:k]:
            hits += 1
    if total == 0:
        return float("nan")
    return hits / total


def mine_hard_negatives(labels, vectors, k: int = 4):
    """Nearest other labels by cosine. Random labels are not used."""
    import numpy as np
    labels = list(labels)
    matrix = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    sims = (matrix / norms) @ (matrix / norms).T
    np.fill_diagonal(sims, -2.0)
    mined = {}
    for i, label in enumerate(labels):
        order = np.argsort(-sims[i])[:k]
        mined[label] = [labels[int(j)] for j in order]
    return mined


def complement_error(p_yes, p_yes_of_negation) -> float:
    return abs(float(p_yes) + float(p_yes_of_negation) - 1.0)


def permutation_report(original: dict, remapped: list) -> dict:
    """original and each remapped item are label -> probability, same label set."""
    import numpy as np
    labels = list(original)
    base = np.array([original[label] for label in labels], dtype=np.float64)
    deltas = []
    flips = 0
    max_abs = 0.0
    for other in remapped:
        alt = np.array([other[label] for label in labels], dtype=np.float64)
        gap = np.abs(base - alt)
        deltas.append(float(gap.mean()))
        max_abs = max(max_abs, float(gap.max()))
        if labels[int(base.argmax())] != labels[int(alt.argmax())]:
            flips += 1
    return {
        "mean_abs_delta": float(np.mean(deltas) if deltas else 0.0),
        "max_abs_delta": max_abs,
        "top1_flip_rate": flips / len(remapped) if remapped else 0.0,
        "n_permutations": len(remapped),
    }


def risk_coverage(correct, confidence) -> list:
    """Error rate at each coverage after dropping the lowest confidence first."""
    import numpy as np
    from veya.common import safe_automation_coverage
    correct = np.asarray(correct, dtype=np.float64)
    confidence = np.asarray(confidence, dtype=np.float64)
    order = np.argsort(-confidence, kind="mergesort")
    correct = correct[order]
    curve = []
    wrong = 0.0
    for i in range(len(correct)):
        wrong += 1.0 - float(correct[i])
        curve.append({"coverage": (i + 1) / len(correct), "error": wrong / (i + 1)})
    sac = {("sac@%.3f" % eps): safe_automation_coverage(correct, confidence, eps)
           for eps in (0.005, 0.01, 0.02, 0.05)}
    return {"curve": curve, "sac": sac}
