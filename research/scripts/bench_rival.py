"""Measured rival head-to-head: Veya x3 vs every available rival on identical bench_apps cases.

Research-only. Reuses bench_apps.build() suites and bench_local's engine/metrics so Veya and
rivals are scored through the SAME metrics() on byte-identical cases.

  USE_TF=0 python3 research/scripts/bench_rival.py [--rival all] [--n 400] \
      [--models-root ~/veya_models] [--device -1]

Requires `datasets` to build the suites (pip install 'veya[train]'). Missing Veya checkpoints and
unavailable rivals are recorded as skips, never fabricated.
"""
import argparse
import gc
import importlib.util
import json
import os
import sys
import time

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HERE = os.path.dirname(os.path.abspath(__file__))          # research/scripts
RESEARCH = os.path.dirname(HERE)                            # research
REPO_ROOT = os.path.dirname(RESEARCH)                       # repo root
for p in (HERE, RESEARCH, REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

import bench_apps  # noqa: E402  (imports bench_local -> veya; paths set above)
import bench_local  # noqa: E402
from veyabench.rivals import REGISTRY  # noqa: E402

VEYA_MODELS = {"english": "veya", "multilingual": "veya-multilingual",
               "typed-decisions": "veya-typed-decisions"}
DEFAULT_OUT = os.path.join(RESEARCH, "results", "rival_benchmark.json")


def _require_datasets():
    if importlib.util.find_spec("datasets") is None:
        sys.stderr.write("bench_rival needs `datasets` to build the suites.\n"
                         "  pip install 'veya[train]'   # or: pip install datasets\n")
        raise SystemExit(2)


def load_veya(name, root):
    import veya
    path = os.path.join(os.path.expanduser(root), VEYA_MODELS[name])
    if not os.path.isdir(path):
        return None, "checkpoint not found: %s" % path
    ag = veya.load(path, device="cpu")
    ag.model.eval()
    return ag, ""


def select_rivals(flag, device=None):
    keys = list(REGISTRY) if flag == "all" else [flag]
    out = []
    for k in keys:
        cls = REGISTRY.get(k)
        if cls is None:
            continue
        r = cls(device=device) if k in ("nli", "st") else cls()
        if r.available():
            out.append(r)
        else:
            print("   skip rival %-12s (%s)" % (r.name, r.unavailable_reason()), flush=True)
    return out


def _veya_metrics(ag, cases, gold):
    lgs, idx, secs, dropped = bench_local.score_cases(ag, cases)
    rows = [(gold[ci], bench_local.softmax_t(z, bench_local.temp_for(ag, qt, k))
             if z is not None else None) for (ci, _, qt, k), z in zip(idx, lgs)]
    m = bench_local.metrics(rows)
    m["seconds"] = round(secs, 1)
    m["ms_per_case"] = round(1000 * secs / max(1, len(cases)), 1)
    m["dropped"] = dropped
    return m


def _rival_metrics(rival, cases, gold):
    rows, secs = rival.score(cases, gold)
    m = bench_local.metrics(rows)
    m["seconds"] = round(secs, 1)
    m["ms_per_case"] = round(1000 * secs / max(1, len(cases)), 1)
    m["dropped"] = int(sum(1 for _, p in rows if p is None))
    return m


def _print_table(suite, block):
    print("\n=== %s ===" % suite, flush=True)
    print("   %-18s %6s %6s %6s %9s" % ("model", "acc", "f1", "ECE", "ms/case"), flush=True)
    for name, m in block.items():
        if "skipped" in m:
            print("   %-18s  -- not measured (%s)" % (name, m["skipped"]), flush=True)
        else:
            print("   %-18s %6.3f %6.3f %6.3f %9.1f"
                  % (name, m["accuracy"], m["macro_f1"], m["ece"], m["ms_per_case"]), flush=True)


def run(args):
    _require_datasets()
    bench_apps.N = args.n                     # build() reads the module global at call time
    print("=== building suites (N=%d per task) ===" % args.n, flush=True)
    bench_apps.build()

    rivals = select_rivals(args.rival, device=args.device)
    results = {"meta": {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "device": str(args.device), "seed": bench_apps.SEED,
                        "n_per_task": args.n, "rivals": [r.name for r in rivals]},
               "suites": {}}

    # Veya checkpoints (loaded once, reused across suites)
    agents, models_present = {}, []
    for mname in VEYA_MODELS:
        ag, why = load_veya(mname, args.models_root)
        if ag is None:
            print("   skip veya/%s (%s)" % (mname, why), flush=True)
        else:
            agents[mname] = ag
            models_present.append(mname)
    results["meta"]["models_present"] = models_present

    for sname, S in bench_apps.SUITES.items():
        cases, gold = S["cases"], S["gold"]
        block = {}
        for mname in models_present:
            try:
                block["veya/" + mname] = _veya_metrics(agents[mname], cases, gold)
            except Exception as e:
                block["veya/" + mname] = {"skipped": str(e)[:120]}
        for rival in rivals:
            try:
                block[rival.name] = _rival_metrics(rival, cases, gold)
            except Exception as e:
                block[rival.name] = {"skipped": str(e)[:120]}
        for r in rivals:
            if r.name == "jev-typesafe":
                block.setdefault(r.name, {"skipped": r.unavailable_reason()})
        if not rivals:
            block.setdefault("jev-typesafe", {"skipped": "no rival selected/available"})
        results["suites"][sname] = block
        _print_table(sname, block)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(results, open(args.out, "w"), indent=2)

    for ag in agents.values():
        del ag
    gc.collect()
    for r in rivals:
        close = getattr(r, "close", None)
        if close:
            close()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print("\nwrote %s" % args.out, flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rival", default="all", choices=["all", "nli", "st", "jev"])
    ap.add_argument("--device", default=None, help="transformers device (e.g. -1 cpu, 0 gpu)")
    ap.add_argument("--models-root", default="~/veya_models")
    ap.add_argument("--n", type=int, default=int(os.environ.get("BENCH_N", "400")))
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args()
    if a.device is not None:
        try:
            a.device = int(a.device)
        except (TypeError, ValueError):
            pass
    run(a)


if __name__ == "__main__":
    main()
