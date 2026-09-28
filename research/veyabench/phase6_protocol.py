"""Phase-6 splits and comparison rules.

The Banking77 public test rows 1300:3080 are the release test. Nothing in this
module reads them. Reranker training, calibration, thresholds, and the
cardinality cutoff use only the training file.
"""
import hashlib

import numpy as np

# Shuffled with random.Random(17), the same shuffle as load_banking("train").
SHORTLIST_TRAIN = (0, 800)
RERANKER_TRAIN = (800, 8600)
CALIBRATION = (8600, 9300)
RERANKER_VAL = (9300, 10003)
# Release test, recorded so callers can refuse it. Not a slice this module returns.
RELEASE_TEST = (1300, 3080)
LOCKED_TEST = (1300, 1500)

# Precommitted before the Phase-6 training run.
SHORTLIST_MIX = 0.8
RANDOM_MIX = 0.2
MAX_EPOCHS = 4
LEARNING_RATE = 1e-5
BATCH_SIZE = 4
HIGH_CONFIDENCE = 0.70


def example_id(split, index):
    if split != "train":
        raise ValueError("phase-6 training ids must come from the train file, got %r" % split)
    return "train:%d" % int(index)


def assert_training_id(example_id_value):
    """Reranker gradient steps may use only the reranker-train slice."""
    if not str(example_id_value).startswith("train:"):
        raise ValueError("refusing non-train example %r" % example_id_value)
    index = int(str(example_id_value).split(":", 1)[1])
    if training_slice_for(index) != "reranker_train":
        raise ValueError("refusing %s example train:%d for a gradient step" % (training_slice_for(index), index))
    return index


def training_slice_for(index):
    """Which phase-6 role a shuffled train index has. Test indices are not accepted."""
    if RERANKER_TRAIN[0] <= index < RERANKER_TRAIN[1]:
        return "reranker_train"
    if CALIBRATION[0] <= index < CALIBRATION[1]:
        return "calibration"
    if RERANKER_VAL[0] <= index < RERANKER_VAL[1]:
        return "reranker_val"
    if SHORTLIST_TRAIN[0] <= index < SHORTLIST_TRAIN[1]:
        return "shortlist_train"
    raise ValueError("train index %d has no phase-6 role" % index)


def split_hash(rows):
    digest = hashlib.sha256()
    for text, label in rows:
        digest.update(b"train\0")
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
        digest.update(label.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def promote(ci_low, high_conf_new, high_conf_base):
    """Keep the simpler head unless validation improvement is unambiguous."""
    if ci_low <= 0:
        return False, "conditional-accuracy interval includes zero; keep the base head"
    if high_conf_new > high_conf_base + 1e-12:
        return False, "high-confidence errors increased"
    return True, "validation conditional accuracy improved without more high-confidence errors"


def paired_bootstrap(correct_base, correct_new, n=1000, seed=17):
    base = np.asarray(correct_base, dtype=np.float64)
    new = np.asarray(correct_new, dtype=np.float64)
    if len(base) != len(new) or len(base) == 0:
        raise ValueError("bootstrap needs two equal non-empty vectors")
    delta = new - base
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(delta), size=(n, len(delta)))
    means = delta[draw].mean(axis=1)
    return {
        "delta": float(delta.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "n": int(len(delta)),
    }


def pair_relation(true_label, pred_label, vectors=None):
    """Coarse relation from the label strings. Residual means none of the rules fired."""
    true_parts = [part for part in true_label.lower().split("_") if part]
    pred_parts = [part for part in pred_label.lower().split("_") if part]
    true_set, pred_set = set(true_parts), set(pred_parts)
    union = true_set | pred_set
    jaccard = (len(true_set & pred_set) / len(union)) if union else 0.0
    same_prefix = bool(true_parts and pred_parts and true_parts[0] == pred_parts[0])
    negation = {"not", "no", "unable", "failed", "wrong", "without"}
    temporal = {"pending", "reverted", "timing", "estimate", "arrival", "expired", "blocked"}
    if same_prefix and jaccard >= 0.25:
        relation = "same_parent"
    elif jaccard >= 0.4:
        relation = "lexical_overlap"
    elif negation & true_set and negation & pred_set and true_set != pred_set:
        relation = "negation_or_failure_pair"
    elif temporal & (true_set | pred_set):
        relation = "temporal_or_status"
    else:
        relation = "residual"
    cosine = None
    if vectors is not None and true_label in vectors and pred_label in vectors:
        a = np.asarray(vectors[true_label], dtype=np.float64)
        b = np.asarray(vectors[pred_label], dtype=np.float64)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        cosine = float(a @ b / denom) if denom else 0.0
        if relation == "residual" and cosine is not None and cosine >= 0.85:
            relation = "embedding_near_neighbor"
    return {"relation": relation, "jaccard": jaccard, "same_prefix": same_prefix, "cosine": cosine}


def confusion_graph(pairs):
    """pairs are (true_label, predicted_label) for reranker mistakes with the target retrieved."""
    edges = {}
    for true_label, pred_label in pairs:
        key = (true_label, pred_label)
        edges[key] = edges.get(key, 0) + 1
    ranked = sorted(edges.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))
    return [{"true": true, "predicted": pred, "count": count} for (true, pred), count in ranked]


def top_confused_partners(edges, label, limit=2):
    partners = [edge for edge in edges if edge["true"] == label]
    partners.sort(key=lambda edge: (-edge["count"], edge["predicted"]))
    return [edge["predicted"] for edge in partners[:limit]]
