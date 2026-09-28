"""Load one Veya release bundle and route Choice without mixing calibrations.

``Agent.load`` still answers packed Choice, Score, and Noul. The learned K=8
projection is Banking77-adapted and is applied only when this bundle is loaded
and the choice router selects the shortlist route. The hierarchical temperature
is never written into the packed ``choice:6-10`` bucket.
"""
import json
import os
import time
from typing import Optional

import numpy as np
import torch

from .common import (
    QTYPES,
    build_sequence,
    collate_items,
    confidence_from_probs,
    serialize_state,
    state_has_content,
    temp_bucket,
)
from .selective import load_policy, score_probabilities
from .shortlist import PRODUCTION_SHORTLIST_ABOVE, PRODUCTION_SHORTLIST_K, _criteria_items
from .shortlist_artifact import ROUTE, file_sha256, load_route_calibration, load_shortlist

ROUTER_FORMAT = "veya-choice-router-v1"
INSTRUCTIONS = "Which label best fits the text?"


def resolve_manifest_path(root: str, value: str) -> str:
    if not isinstance(value, str) or not value or os.path.isabs(value):
        raise ValueError("manifest paths must be relative, got %r" % (value,))
    root_path = os.path.realpath(root)
    resolved = os.path.realpath(os.path.join(root_path, value))
    if os.path.commonpath((root_path, resolved)) != root_path:
        raise ValueError("manifest path escapes the release root: %r" % (value,))
    return resolved


def load_router(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        router = json.load(handle)
    if router.get("format") != ROUTER_FORMAT:
        raise ValueError("unsupported choice router format %r" % router.get("format"))
    if int(router.get("hierarchical_k", -1)) != PRODUCTION_SHORTLIST_K:
        raise ValueError("router K does not match the production shortlist K")
    if router.get("hierarchical_route") != ROUTE:
        raise ValueError("router route %r is not %s" % (router.get("hierarchical_route"), ROUTE))
    if int(router.get("packed_max_candidates", -1)) != PRODUCTION_SHORTLIST_ABOVE:
        raise ValueError("router cutoff does not match production_choice_plan")
    if int(router.get("embed_max_length", 0)) < 1:
        raise ValueError("router is missing embed_max_length")
    return router


def recommend_packed_cutoff(gap_by_n, grid, floor=0.05, current=32):
    """Largest packed cardinality still supported by a measured accuracy gap.

    ``gap_by_n`` is hierarchical accuracy minus packed accuracy. A cardinality
    counts as a hierarchical win only when its gap is at least ``floor`` and
    every larger grid point also clears that floor. The cutoff is the grid
    point just below the first such win. A weak 77-way gap keeps ``current``.
    """
    if gap_by_n.get(77, 0.0) < floor:
        return current, "77-way gap is below %.2f on this sweep; keep the existing cutoff" % floor
    winning = []
    for n in grid:
        if gap_by_n.get(n, 0.0) < floor:
            continue
        if all(gap_by_n.get(m, 0.0) >= floor for m in grid if m >= n):
            winning.append(n)
    if not winning:
        return current, "no grid point stays above the %.2f gap through 77" % floor
    first = min(winning)
    below = [n for n in grid if n < first]
    if not below:
        return 1, "hierarchical leads by >= %.2f at every measured cardinality" % floor
    return below[-1], "hierarchical leads by >= %.2f from %d candidates upward" % (floor, first)


def plan_for(router: dict, n_options: int) -> dict:
    if isinstance(n_options, bool) or not isinstance(n_options, int) or n_options < 1:
        raise ValueError("n_options must be a positive integer")
    if n_options <= int(router["packed_max_candidates"]):
        return {"path": "packed", "k": None, "calibration": "packed"}
    return {"path": "shortlist", "k": int(router["hierarchical_k"]), "calibration": ROUTE}


def normalize_criteria(criteria):
    """Return validated criteria without silently collapsing duplicate list labels."""
    return _criteria_items(criteria)


def _softmax(logits, temperature: float) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / float(temperature)
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()


def raw_choice_logits(agent, state, instructions: str, criteria: dict, max_len=None, head_max_len=None, option_token_cap: int = 48):
    """Pre-temperature choice logits. One row, one column per criterion, in dict order."""
    question = {"t": "choice", "ins": instructions, "crit": criteria}
    max_len = int(agent.cfg.get("max_len", 512) if max_len is None else max_len)
    head_max_len = int(agent.cfg.get("head_max_len", 192) if head_max_len is None else head_max_len)
    seq, markers = build_sequence(
        agent.tok, state, question, max_len, head_max_len, option_token_cap=int(option_token_cap),
    )
    if len(markers) != len(criteria):
        raise ValueError("packed head kept %d of %d options" % (len(markers), len(criteria)))
    items = [{"ids": seq, "markers": markers, "qtype": QTYPES["choice"]}]
    batch = collate_items([items], agent.tok.pad_token_id)
    with torch.inference_mode():
        logits, _act = agent.model(
            batch["input_ids"].to(agent.device),
            batch["attention_mask"].to(agent.device),
            batch["marker_pos"].to(agent.device),
            batch["marker_mask"].to(agent.device),
            batch["qtype"].to(agent.device),
        )
    return logits[0, : len(criteria)].float().cpu().numpy(), int(batch["attention_mask"].sum())


class ReleaseBundle:
    def __init__(self, agent, router, projection, shortlist_blob, route_calibration, checkpoint_hash, shortlist_sha256):
        self.agent = agent
        self.router = router
        self.projection = projection
        self.projection.eval()
        self.labels = list(shortlist_blob["labels"])
        self.label_texts = list(shortlist_blob["label_texts"])
        self.route_calibration = route_calibration
        self.route_temperature = float(route_calibration["temperature"])
        self.checkpoint_hash = checkpoint_hash
        self.shortlist_sha256 = shortlist_sha256
        self.selective_policy = None
        self.reranker = None
        self.reranker_id = "base_decision_head"
        self.reranker_sha256 = None
        self.hypothesis = None
        self.manifest: Optional[dict] = None
        self.manifest_path: Optional[str] = None
        self.manifest_sha256: Optional[str] = None
        self._embed = None
        self._cache = {}

    def plan(self, n_options: int) -> dict:
        return plan_for(self.router, n_options)

    def embed(self, texts):
        if self._embed is None:
            from .shortlist import embed_fn_from_agent
            self._embed = embed_fn_from_agent(
                self.agent,
                max_length=int(self.router["embed_max_length"]),
                batch_size=16,
            )
        missing = [text for text in texts if text not in self._cache]
        if missing:
            vectors = self._embed(missing)
            for text, vector in zip(missing, vectors):
                self._cache[text] = np.asarray(vector, dtype=np.float32)
        return np.stack([self._cache[text] for text in texts], axis=0)

    def project(self, vectors: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self.projection(torch.tensor(vectors, dtype=torch.float32)).numpy()

    def shortlist(self, state, criteria_items, k: int):
        """Projection dot-product. The query is the state text, as in training."""
        keys = [key for key, _value in criteria_items]
        if k >= len(keys):
            return list(keys), {"state_embed": 0.0, "candidate_embed": 0.0, "shortlist_score": 0.0}
        query = serialize_state(state)
        cand = []
        for key, value in criteria_items:
            if isinstance(value, str) and value != "":
                cand.append(value)
            else:
                cand.append(str(key))
        started = time.perf_counter()
        qv = self.project(self.embed([query]))
        state_s = time.perf_counter() - started
        started = time.perf_counter()
        cv = self.project(self.embed(cand))
        cand_s = time.perf_counter() - started
        started = time.perf_counter()
        scores = (qv @ cv.T).reshape(-1)
        top = torch.tensor(scores).topk(int(k)).indices.tolist()
        score_s = time.perf_counter() - started
        return [keys[i] for i in top], {
            "state_embed": state_s,
            "candidate_embed": cand_s,
            "shortlist_score": score_s,
        }

    def temperature_for(self, path: str, k: int) -> float:
        if path == "packed":
            bucket = temp_bucket(QTYPES["choice"], k)
            return float(self.agent.temperature_by_options.get(bucket, self.agent.temperature[0]))
        if path == "shortlist":
            return self.route_temperature
        raise ValueError("unknown choice path %r" % path)

    def score_choice(self, state, instructions: str, criteria, timings: bool = False):
        items = normalize_criteria(criteria)
        crit = {key: value for key, value in items}
        plan = self.plan(len(items))
        stages = {}
        if plan["path"] == "shortlist":
            picked, stage = self.shortlist(state, items, plan["k"])
            stages.update(stage)
            crit = {label: crit[label] for label in picked}
            shortlist = list(picked)
            path = "shortlist"
        else:
            shortlist = None
            path = "packed"
        started = time.perf_counter()
        route_name = ROUTE if path == "shortlist" else "packed"
        if path == "shortlist" and self.hypothesis is not None:
            from .experimental.hypothesis import texts_for_labels
            texts = texts_for_labels(list(crit.keys()), self.hypothesis.variant)
            logits_t, n_tokens, _truncated = self.hypothesis.logits_for(state, instructions, texts)
            logits = logits_t.detach().float().cpu().numpy()
            temperature = float(self.hypothesis.temperature)
            route_name = "shortlist_k8_hypothesis_hce2"
            plan = dict(plan)
            plan["calibration"] = self.hypothesis.calibration_id
        elif path == "shortlist" and self.reranker is not None:
            logits, n_tokens = self.reranker.choice_logits(self.agent, state, instructions, crit)
            temperature = self.temperature_for(path, len(crit))
        else:
            logits, n_tokens = raw_choice_logits(self.agent, state, instructions, crit)
            temperature = self.temperature_for(path, len(crit))
        stages["reranker"] = time.perf_counter() - started
        started = time.perf_counter()
        probs = _softmax(logits, temperature)
        stages["calibration"] = time.perf_counter() - started
        labels = list(crit.keys())
        pred = labels[int(np.argmax(probs))]
        out = {
            "path": path,
            "route": route_name,
            "calibration": plan["calibration"],
            "temperature": temperature,
            "labels": labels,
            "shortlist": shortlist,
            "probabilities": probs,
            "prediction": pred,
            "logits": logits,
            "input_tokens": n_tokens,
            "stages_s": stages if timings else None,
        }
        return out

    def public_choice(self, state, instructions: str, criteria) -> dict:
        from .lang import defer_reason
        if not state_has_content(state):
            return {
                "model": "veya-rl-agent", "decision": "defer", "reason": "invalid_schema",
                "route": None, "primitive": "choice", "answer": None, "probabilities": None,
                "confidence": None, "max_probability": None, "automation": None,
                "deferred": True, "defer_reason": "invalid_schema", "shortlist": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            }
        blocked = defer_reason(self.agent.cfg, state)
        if blocked:
            return {
                "model": "veya-rl-agent",
                "decision": "defer",
                "reason": blocked,
                "route": None,
                "primitive": "choice",
                "answer": None,
                "probabilities": None,
                "confidence": None,
                "max_probability": None,
                "automation": None,
                "deferred": True,
                "defer_reason": blocked,
                "shortlist": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            }
        scored = self.score_choice(state, instructions, criteria, timings=False)
        probs = scored["probabilities"]
        labels = scored["labels"]
        k = len(labels)
        automation = None
        reason = None
        decision = "answer"
        if scored["path"] == "shortlist" and self.selective_policy:
            kind = self.selective_policy["score"]
            threshold = self.selective_policy["thresholds"].get(self.selective_policy["operating_epsilon"])
            value = score_probabilities(probs, kind)
            if threshold is None or value < float(threshold) - 1e-12:
                automation = "defer"
            else:
                automation = "answer"
        payload = {
            "model": "veya-rl-agent",
            "decision": decision,
            "reason": reason,
            "route": scored["route"],
            "calibration": scored["calibration"],
            "temperature": scored["temperature"],
            "primitive": "choice",
            "answer": scored["prediction"],
            "probabilities": {label: round(float(p), 4) for label, p in zip(labels, probs)},
            "confidence": round(confidence_from_probs(probs, k), 4),
            "max_probability": round(float(probs.max()), 4),
            "automation": automation,
            "deferred": False,
            "defer_reason": None,
            "shortlist": scored["shortlist"],
            "reranker_id": self.reranker_id if scored["path"] == "shortlist" else "base_decision_head",
            "candidate_count": k if scored["shortlist"] is None else len(normalize_criteria(criteria)),
            "usage": {"input_tokens": scored["input_tokens"], "output_tokens": 0},
        }
        return payload

    def answer_questions(self, state, questions: dict) -> dict:
        """Route each choice. High-N uses the attached hypothesis scorer. Score and noul stay packed."""
        from .lang import defer_reason
        if not isinstance(questions, dict) or not questions or not state_has_content(state):
            return {
                "model": "veya-rl-agent",
                "decision": "defer",
                "reason": "invalid_schema",
                "answers": {},
                "routes": {},
                "usage": {"input_tokens": 0, "output_tokens": 0},
            }
        blocked = defer_reason(self.agent.cfg, state)
        if blocked:
            return {
                "model": "veya-rl-agent",
                "decision": "defer",
                "reason": blocked,
                "answers": {},
                "routes": {},
                "usage": {"input_tokens": 0, "output_tokens": 0},
            }
        packed = {}
        routes = {}
        answers = {}
        tokens = 0
        for qid, question in questions.items():
            internal = self.agent._to_internal(question)
            kind = question.get("type")
            if kind != "choice":
                packed[qid] = question
                routes[qid] = kind
                continue
            criteria = internal["crit"]
            n = len(normalize_criteria(criteria))
            plan = self.plan(n)
            if plan["path"] == "packed":
                packed[qid] = question
                routes[qid] = "packed"
            else:
                public = self.public_choice(state, question.get("instructions") or "", criteria)
                answers[qid] = {
                    "type": "choice",
                    "choice": public["answer"],
                    "probabilities": public["probabilities"],
                    "confidence": public["confidence"],
                    "max_probability": public["max_probability"],
                    "route": public["route"],
                    "automation": public["automation"],
                    "shortlist": public["shortlist"],
                }
                routes[qid] = public["route"]
                tokens += public["usage"]["input_tokens"]
        if packed:
            result = self.agent.system_one(state, packed)
            if result.get("decision") != "answer":
                return result
            answers.update(result.get("answers") or {})
            tokens += int((result.get("usage") or {}).get("input_tokens") or 0)
        return {
            "model": "veya-rl-agent",
            "decision": "answer",
            "reason": None,
            "answers": answers,
            "routes": routes,
            "calibration_version": getattr(self.agent, "calibration_version", "checkpoint"),
            "usage": {"input_tokens": tokens, "output_tokens": 0},
        }


def load_bundle(model_dir: str, router_path: str, shortlist_path: str, route_calibration_path: str,
                checkpoint_hash: str, device: Optional[str] = None,
                timings: Optional[dict] = None) -> ReleaseBundle:
    """Load checkpoint, shortlist, and route calibration. Refuses mismatched hashes."""
    import veya

    def mark(name, started):
        if timings is not None:
            timings[name] = time.perf_counter() - started
        return time.perf_counter()

    started = time.perf_counter()
    weights = os.path.join(model_dir, "model.safetensors")
    digest = file_sha256(weights)
    if digest != checkpoint_hash:
        raise ValueError("checkpoint hash does not match the manifest")
    started = mark("checkpoint_hash_s", started)
    agent = veya.load(model_dir, device=device)
    agent.model.eval()
    started = mark("checkpoint_load_s", started)
    router = load_router(router_path)
    projection, blob = load_shortlist(shortlist_path, digest, k=PRODUCTION_SHORTLIST_K)
    shortlist_sha = file_sha256(shortlist_path)
    if blob.get("checkpoint_hash") != digest:
        raise ValueError("shortlist artifact is bound to a different checkpoint")
    started = mark("shortlist_load_s", started)
    route = load_route_calibration(route_calibration_path, digest, shortlist_sha)
    if int(route.get("k", -1)) != PRODUCTION_SHORTLIST_K:
        raise ValueError("route calibration K does not match the shortlist")
    mark("route_calibration_load_s", started)
    return ReleaseBundle(agent, router, projection, blob, route, digest, shortlist_sha)


def load_release_manifest(manifest_path: str, device: Optional[str] = None,
                          root: Optional[str] = None, timings: Optional[dict] = None):
    """Load and verify every component named by a release manifest.

    This is the production startup path. It validates file hashes before attaching
    calibration and policy artifacts, and refuses paths outside ``root``.
    """
    manifest_path = os.path.realpath(manifest_path)
    if root is None:
        root = os.path.dirname(os.path.dirname(os.path.dirname(manifest_path)))
    root = os.path.realpath(root)
    if os.path.commonpath((root, manifest_path)) != root:
        raise ValueError("release manifest is outside the release root")
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("format") != "veya-release-manifest-v2":
        raise ValueError("unsupported release manifest format %r" % manifest.get("format"))

    def verified_path(value, expected_hash, name):
        path = resolve_manifest_path(root, value)
        if not os.path.isfile(path):
            raise ValueError("%s file is missing: %s" % (name, value))
        actual = file_sha256(path)
        if actual != expected_hash:
            raise ValueError("%s hash does not match the release manifest" % name)
        return path

    for relative, expected_hash in (manifest.get("runtime_files") or {}).items():
        verified_path(relative, expected_hash, "runtime")

    model_dir = resolve_manifest_path(root, manifest["base_checkpoint"])
    config_path = verified_path(manifest["model_config"], manifest["model_config_hash"], "model config")
    expected_config = os.path.realpath(os.path.join(model_dir, "rl_agent_config.json"))
    if config_path != expected_config:
        raise ValueError("model config is not inside the selected base checkpoint")
    for relative, expected_hash in (manifest.get("tokenizer_files") or {}).items():
        verified_path(relative, expected_hash, "tokenizer")
    router_path = verified_path(
        manifest["routing"]["choice_router"], manifest["routing"]["choice_router_hash"], "choice router",
    )
    shortlist_path = verified_path(
        manifest["choice"]["shortlist_artifact"], manifest["choice"]["shortlist_hash"], "shortlist",
    )
    packed_calibration_path = verified_path(
        manifest["calibration"]["packed_choice"], manifest["calibration"]["packed_choice_hash"],
        "packed calibration",
    )
    route_calibration_path = verified_path(
        manifest["calibration"]["hierarchical_choice"],
        manifest["calibration"]["hierarchical_choice_hash"], "route calibration",
    )
    bundle = load_bundle(
        model_dir, router_path, shortlist_path, route_calibration_path,
        manifest["base_checkpoint_hash"], device=device, timings=timings,
    )
    from .calibration import load_calibration
    load_calibration(packed_calibration_path, checkpoint_hash=bundle.checkpoint_hash)
    if int(bundle.router["packed_max_candidates"]) != int(manifest["choice"]["packed_max_candidates"]):
        raise ValueError("choice router cutoff does not match the release manifest")
    if int(bundle.router["hierarchical_k"]) != int(manifest["choice"]["shortlist_k"]):
        raise ValueError("choice router K does not match the release manifest")
    if abs(bundle.route_temperature - float(manifest["calibration"]["hierarchical_temperature"])) > 1e-12:
        raise ValueError("route calibration temperature does not match the release manifest")

    policy_path = verified_path(
        manifest["selective_policy"], manifest["selective_policy_hash"], "selective policy",
    )
    bundle.selective_policy = load_policy(
        policy_path, ROUTE, bundle.checkpoint_hash, bundle.shortlist_sha256,
        bundle.reranker_id,
    )
    bundle.manifest = manifest
    bundle.manifest_path = manifest_path
    bundle.manifest_sha256 = file_sha256(manifest_path)
    return bundle, manifest
