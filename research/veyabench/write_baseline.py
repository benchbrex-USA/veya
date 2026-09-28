"""Write research/results/post_fix_baseline.json from the local English checkpoint.

Cardinality uses the tokenizer only. The live section loads the English weights once.
"""
import json
import os
import platform
import random
import subprocess
import time

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL = os.path.join(ROOT, "models", "production")
OUT = os.path.join(ROOT, "research", "results", "post_fix_baseline.json")


def commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def cardinality():
    from transformers import AutoTokenizer
    from veya.common import build_sequence
    tok = AutoTokenizer.from_pretrained(os.path.join(MODEL, "tokenizer"))
    state = "Please refund invoice 4411. " * 40
    rows = []
    for n in (2, 10, 20, 50, 100, 150, 255):
        crit = {("opt%03d" % i): "category %d" % i for i in range(n)}
        q = {"t": "choice", "ins": "Which category fits the message?", "crit": crit}
        ids, markers = build_sequence(tok, {"message": state}, q, max_len=512, head_max_len=192)
        rows.append({"options": n, "sequence_tokens": len(ids), "markers_kept": len(markers),
                     "dropped": n - len(markers)})
    return rows


def live():
    import veya
    from research.veyabench.protocol import permutation_report
    started = time.time()
    agent = veya.load(MODEL, device="cpu")
    question = {"dept": {"type": "choice", "instructions": "Which team handles `message`?",
                         "criteria": {"billing": "invoices and refunds", "technical": "bugs and outages",
                                      "sales": "pricing and new purchases", "hr": "hiring and payroll"}}}
    state = {"message": "Please refund invoice 4411. I was charged twice."}
    base = agent.predict(state, question)
    probs = base["answers"]["dept"]["probabilities"]
    labels = list(question["dept"]["criteria"])
    remapped = []
    rng = random.Random(0)
    for _ in range(8):
        order = labels[:]
        rng.shuffle(order)
        crit = {key: question["dept"]["criteria"][key] for key in order}
        q = {"dept": {"type": "choice", "instructions": question["dept"]["instructions"], "criteria": crit}}
        got = agent.predict(state, q)["answers"]["dept"]["probabilities"]
        remapped.append({key: got[key] for key in labels})
    hindi = agent.predict({"message": "मुझसे इनवॉइस 4411 के लिए दो बार शुल्क लिया गया"}, question)
    empty = agent.predict(state, {})
    return {
        "load_seconds": round(time.time() - started, 2),
        "device": str(agent.device),
        "calibration_version": agent.calibration_version,
        "english_choice": base["answers"]["dept"]["choice"],
        "english_probabilities": probs,
        "hindi": {"decision": hindi["decision"], "reason": hindi["reason"]},
        "empty_questions": {"decision": empty["decision"], "reason": empty["reason"]},
        "permutation": permutation_report(probs, remapped),
    }


def main():
    import torch
    report = {
        "commit": commit(),
        "checkpoint": MODEL,
        "hardware": platform.platform(),
        "torch": torch.__version__,
        "results": {
            "cardinality": cardinality(),
            "language": {},
            "choice": {},
            "calibration": {"runtime": "checkpoint", "note": "no held-out calibration.json fitted"},
        },
    }
    report["results"].update(live())
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
