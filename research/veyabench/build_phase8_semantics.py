"""Training-side candidate semantics for Banking77.

Official PolyAI categories.json is a name list. This file does not invent
definitions from label names, and it does not read the release test.
Each description quotes one training utterance and records that row.
"""
import json
import os
import re

from research.veyabench.phase6_protocol import RERANKER_TRAIN
from research.veyabench.run_phase3_eval import ROOT, load_banking

ART = os.path.join(ROOT, "artifacts", "experiments", "phase8")
SOURCE = "PolyAI-LDN/task-specific-datasets banking_data/categories.json names; examples from local train.csv seed 17"
WORDS = re.compile(r"[a-z0-9']+")


def clip_words(text, n):
    return " ".join(text.replace("\n", " ").split()[:n])


def first_hit(rows, label, start, end):
    for index in range(start, end):
        if rows[index][1] == label:
            return index, rows[index][0]
    return None, None


def tokens(text):
    return set(WORDS.findall(text.lower()))


def main():
    os.makedirs(ART, exist_ok=True)
    train = load_banking("train")
    labels = sorted({row[1] for row in train})
    edges = json.load(open(os.path.join(ROOT, "artifacts", "experiments", "phase6", "train_confusion.json"), encoding="utf-8"))["top_edges"]
    partner = {}
    for edge in edges:
        partner.setdefault(edge["true"], edge["predicted"])
    start, end = RERANKER_TRAIN
    records = []
    missing = []
    for label in labels:
        index, text = first_hit(train, label, start, end)
        source_slice = "reranker_train_800_8600"
        if index is None:
            index, text = first_hit(train, label, 0, start)
            source_slice = "shortlist_train_0_800"
        if index is None:
            missing.append(label)
            continue
        other = partner.get(label)
        other_index, other_text = (None, None)
        if other:
            other_index, other_text = first_hit(train, other, start, end)
        pretty = label.replace("_", " ")
        example = clip_words(text, 10)
        definition = "Customers ask things like: %s" % example
        boundary = None
        if other and other_text:
            boundary = "Different from %s, for example: %s" % (other.replace("_", " "), clip_words(other_text, 8))
        records.append({
            "schema": "banking77",
            "candidate_id": label,
            "name": pretty,
            "definition": definition,
            "does_not_apply_when": boundary,
            "example_id": "train:%d" % index,
            "contrast_example_id": None if other_index is None else "train:%d" % other_index,
            "contrast_label": other,
            "source": SOURCE,
            "source_slice": source_slice,
            "generation_method": "first_training_utterance_in_slice",
            "manually_written": False,
            "model_generated": False,
            "review_status": "automatic_unreviewed",
            "official_definition": None,
        })
    ambiguous = []
    by_id = {row["candidate_id"]: row for row in records}
    for left in records:
        for right_id, right in by_id.items():
            if right_id <= left["candidate_id"]:
                continue
            a = tokens(left["definition"])
            b = tokens(right["definition"])
            union = a | b
            if union and len(a & b) / len(union) >= 0.6:
                ambiguous.append([left["candidate_id"], right_id])
    payload = {
        "schema": "banking77",
        "version": "phase8-train-utterance-v1",
        "official_definition_available": False,
        "official_source": "https://github.com/PolyAI-LDN/task-specific-datasets/blob/master/banking_data/categories.json",
        "official_source_contents": "names only",
        "release_test_used": False,
        "records": records,
        "missing_training_example": missing,
        "schema_ambiguous_pairs": ambiguous,
    }
    path = os.path.join(ART, "candidate_semantics.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print("labels", len(records), "missing", len(missing), "ambiguous_pairs", len(ambiguous))


if __name__ == "__main__":
    main()
