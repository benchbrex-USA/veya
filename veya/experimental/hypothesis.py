"""Independent state-hypothesis compatibility scores.

Each candidate is encoded as its own pair. Nothing in the pair says where that
candidate sat in the list. The only set-level step is a softmax over the K
scalars. This module does not replace packed Choice.
"""
import copy
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from veya.common import serialize_state

FORMAT = "veya-hypothesis-scorer-v1"
TEMPLATE_VERSION = "phase9-h0-v1"
# Premise side is truncated. The hypothesis side, including its mask, is kept whole.
MAX_LENGTH = 160


def hypothesis_text(label, variant="H0", definition=None):
    """One candidate as a complete hypothesis. H2 refuses invented or contrastive text."""
    name = str(label).strip()
    if not name:
        raise ValueError("hypothesis label is empty")
    if variant == "H0":
        return "The correct category is: %s." % name
    if variant == "H1":
        return "The correct category is a %s." % name.replace("_", " ")
    if variant == "H2":
        text = "" if definition is None else str(definition).strip()
        if not text:
            raise ValueError("H2 requires an authoritative definition")
        lowered = text.lower()
        if "unlike" in lowered or "different from" in lowered:
            raise ValueError("H2 refuses contrastive boundary text")
        return "The correct category is: %s. %s" % (name, text)
    raise ValueError("unknown hypothesis variant %r" % variant)


def pair_segments(state, question, hypothesis):
    """Premise is state plus the decision question. The hypothesis is the other segment."""
    premise = "[STATE]\n%s\n\n[QUESTION]\n%s" % (serialize_state(state), str(question))
    second = "[HYPOTHESIS]\n%s" % str(hypothesis)
    if str(hypothesis) and str(hypothesis) in premise:
        raise ValueError("hypothesis text leaked into the premise")
    return premise, second


def validate_hypotheses(hypotheses):
    """Reject duplicate ids and identical hypothesis strings before scoring."""
    if not hypotheses:
        raise ValueError("at least one hypothesis is required")
    seen = set()
    for text in hypotheses:
        key = " ".join(str(text).split()).casefold()
        if not key:
            raise ValueError("empty hypothesis")
        if key in seen:
            raise ValueError("duplicate hypothesis text")
        seen.add(key)
    return list(hypotheses)


def _sanitize(tokenizer, text):
    mask = tokenizer.mask_token or ""
    if mask:
        return str(text).replace(mask, " ")
    return str(text)


def encode_pairs(tokenizer, premises, hypotheses, max_length=MAX_LENGTH):
    """[CLS] premise [SEP] [MASK] hypothesis [SEP]. The mask is the compatibility position.

    Premise tokens are the only ones dropped when the budget is tight. Padding is
    on the right and is covered by the attention mask.
    """
    if len(premises) != len(hypotheses):
        raise ValueError("premise and hypothesis counts differ")
    rows = []
    markers = []
    truncated = 0
    for premise, hypothesis in zip(premises, hypotheses):
        pre = tokenizer(_sanitize(tokenizer, premise), add_special_tokens=False)["input_ids"]
        hyp = tokenizer(_sanitize(tokenizer, hypothesis), add_special_tokens=False)["input_ids"]
        budget = int(max_length) - 3 - len(hyp)
        if budget < 8:
            raise ValueError("hypothesis does not fit in the pair budget")
        if len(pre) > budget:
            pre = pre[:budget]
            truncated += 1
        marker = 1 + len(pre) + 1
        ids = [tokenizer.cls_token_id] + pre + [tokenizer.sep_token_id, tokenizer.mask_token_id] + hyp + [tokenizer.sep_token_id]
        if ids[marker] != tokenizer.mask_token_id:
            raise RuntimeError("compatibility mask is not where the pair builder put it")
        rows.append(ids)
        markers.append(marker)
    width = max(len(row) for row in rows)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("tokenizer has no pad token")
    input_ids = torch.full((len(rows), width), int(pad_id), dtype=torch.long)
    attention = torch.zeros((len(rows), width), dtype=torch.long)
    for index, row in enumerate(rows):
        input_ids[index, : len(row)] = torch.tensor(row, dtype=torch.long)
        attention[index, : len(row)] = 1
    return input_ids, attention, torch.tensor(markers, dtype=torch.long), truncated


class CompatibilityHead(nn.Module):
    """Linear scalar on one hidden state. LayerNorm keeps the init on a stable scale.

    The forward takes only that vector. There is no candidate-position feature.
    """

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(int(dim))
        self.score = nn.Linear(int(dim), 1)
        nn.init.normal_(self.score.weight, std=0.02)
        nn.init.zeros_(self.score.bias)

    def forward(self, hidden):
        return self.score(self.norm(hidden.float())).squeeze(-1)


def choice_probabilities(logits, temperature):
    """Softmax of logits divided by a positive temperature. Temperature does not change argmax."""
    if temperature is None or float(temperature) <= 0:
        raise ValueError("choice temperature must be positive")
    values = np.asarray(logits, dtype=np.float64) / float(temperature)
    values = values - np.max(values)
    probs = np.exp(values)
    return probs / probs.sum()


def encode_private_suffix(base, private_layers, private_norm, input_ids, attention_mask, start_layer):
    """Base prefix stays on the shared encoder. The adapted suffix is a private copy."""
    from transformers.masking_utils import create_bidirectional_mask, create_bidirectional_sliding_window_mask

    position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
    with torch.no_grad():
        hidden = base.embeddings(input_ids=input_ids)
        mask_kwargs = {"config": base.config, "inputs_embeds": hidden, "attention_mask": attention_mask}
        masks = {
            "full_attention": create_bidirectional_mask(**mask_kwargs),
            "sliding_attention": create_bidirectional_sliding_window_mask(**mask_kwargs),
        }
        positions = {}
        for layer_type in set(base.config.layer_types):
            positions[layer_type] = base.rotary_emb(hidden, position_ids, layer_type)
        for layer in base.layers[:start_layer]:
            hidden = layer(
                hidden,
                attention_mask=masks[layer.attention_type],
                position_embeddings=positions[layer.attention_type],
            )
        hidden = hidden.detach()
        for offset, layer in enumerate(private_layers):
            base_layer = base.layers[start_layer + offset]
            hidden = layer(
                hidden,
                attention_mask=masks[base_layer.attention_type],
                position_embeddings=positions[base_layer.attention_type],
            )
        hidden = private_norm(hidden)
    return hidden


class IsolatedAdaptedEncoder(nn.Module):
    """Upper-layer delta that does not write into the shared production encoder."""

    def __init__(self, base_encoder, delta):
        super().__init__()
        start = int(delta["start_layer"])
        if start < 0 or start >= len(base_encoder.layers):
            raise ValueError("encoder delta start_layer %s is outside the encoder" % start)
        if len(delta["layers"]) != len(base_encoder.layers) - start:
            raise ValueError("encoder delta does not cover the suffix it names")
        object.__setattr__(self, "_base", base_encoder)
        self.start_layer = start
        copied = []
        for offset, state in enumerate(delta["layers"]):
            layer = copy.deepcopy(base_encoder.layers[start + offset])
            layer.load_state_dict(state)
            copied.append(layer)
        self.suffix = nn.ModuleList(copied)
        self.final_norm = copy.deepcopy(base_encoder.final_norm)
        self.final_norm.load_state_dict(delta["final_norm"])
        self.eval()

    def forward(self, input_ids=None, attention_mask=None, **_kwargs):
        hidden = encode_private_suffix(
            self._base, self.suffix, self.final_norm, input_ids, attention_mask, self.start_layer,
        )
        return SimpleNamespace(last_hidden_state=hidden)


def encode_suffix(encoder, input_ids, attention_mask, start_layer):
    """Frozen prefix under no_grad. Gradients start at start_layer. Matches ModernBERT."""
    from transformers.masking_utils import create_bidirectional_mask, create_bidirectional_sliding_window_mask

    position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
    with torch.no_grad():
        hidden = encoder.embeddings(input_ids=input_ids)
        mask_kwargs = {"config": encoder.config, "inputs_embeds": hidden, "attention_mask": attention_mask}
        masks = {
            "full_attention": create_bidirectional_mask(**mask_kwargs),
            "sliding_attention": create_bidirectional_sliding_window_mask(**mask_kwargs),
        }
        positions = {}
        for layer_type in set(encoder.config.layer_types):
            positions[layer_type] = encoder.rotary_emb(hidden, position_ids, layer_type)
        for layer in encoder.layers[:start_layer]:
            hidden = layer(
                hidden,
                attention_mask=masks[layer.attention_type],
                position_embeddings=positions[layer.attention_type],
            )
        hidden = hidden.detach()
    for layer in encoder.layers[start_layer:]:
        hidden = layer(
            hidden,
            attention_mask=masks[layer.attention_type],
            position_embeddings=positions[layer.attention_type],
        )
    return encoder.final_norm(hidden)


def compatibility_logits(encoder, head, input_ids, attention_mask, marker_pos, device, grad_from_layer=None):
    """One scalar per row. grad_from_layer None freezes the whole encoder."""
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    marker_pos = marker_pos.to(device)
    if grad_from_layer is None:
        with torch.no_grad():
            hidden = encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    else:
        hidden = encode_suffix(encoder, input_ids, attention_mask, int(grad_from_layer))
    index = torch.arange(hidden.size(0), device=hidden.device)
    return head(hidden[index, marker_pos])


def score_groups(encoder, head, tokenizer, states, question, hypothesis_lists, device, pair_batch=32,
                 max_length=MAX_LENGTH, grad_from_layer=None):
    """Flatten groups, encode them in batches, and return one logit tensor per group.

    A group is a list of hypothesis strings. Scores follow that list order and do
    not depend on any other group's contents.
    """
    premises = []
    seconds = []
    widths = []
    for state, hypotheses in zip(states, hypothesis_lists):
        validate_hypotheses(hypotheses)
        for text in hypotheses:
            premise, second = pair_segments(state, question, text)
            premises.append(premise)
            seconds.append(second)
        widths.append(len(hypotheses))
    pieces = []
    token_rows = []
    truncated = 0
    for start in range(0, len(premises), int(pair_batch)):
        stop = start + int(pair_batch)
        input_ids, attention, markers, cut = encode_pairs(tokenizer, premises[start:stop], seconds[start:stop], max_length)
        truncated += cut
        pieces.append(compatibility_logits(encoder, head, input_ids, attention, markers, device, grad_from_layer))
        token_rows.extend(int(value) for value in attention.sum(dim=1).tolist())
    flat = torch.cat(pieces, dim=0) if pieces else torch.zeros(0, device=device)
    groups = []
    tokens = []
    cursor = 0
    for width in widths:
        groups.append(flat[cursor:cursor + width])
        tokens.append(int(sum(token_rows[cursor:cursor + width])))
        cursor += width
    return groups, tokens, truncated


def texts_for_labels(labels, variant="H0", definitions=None):
    definitions = definitions or {}
    return [hypothesis_text(label, variant, definitions.get(label)) for label in labels]


def validate_phase9_record(record, checkpoint_hash, shortlist_sha256, head_sha256, delta_sha256):
    """Refuse a scorer whose recorded identity does not match the files being loaded."""
    required = (
        "primitive", "route", "k", "method", "temperature", "fitting_split", "sample_count",
        "checkpoint_hash", "shortlist_sha256", "head_sha256", "encoder_delta_sha256",
        "template_version", "start_layer",
    )
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError("phase-9 calibration is missing %s" % ", ".join(missing))
    if record["primitive"] != "choice" or int(record["k"]) != 8:
        raise ValueError("phase-9 calibration is not K=8 Choice")
    if record["route"] != "hypothesis_hce2_h0":
        raise ValueError("phase-9 calibration route does not match the scorer")
    if record["method"] not in ("temperature_scaling", "temperature-nll"):
        raise ValueError("phase-9 calibration method is not temperature scaling")
    if float(record["temperature"]) <= 0:
        raise ValueError("phase-9 temperature must be positive")
    if record["fitting_split"] in (None, "", "test", "release_test", "reranker_val"):
        raise ValueError("phase-9 calibration split %r is not an allowed fit split" % record["fitting_split"])
    if record["checkpoint_hash"] != checkpoint_hash:
        raise ValueError("phase-9 calibration belongs to a different checkpoint")
    if record["shortlist_sha256"] != shortlist_sha256:
        raise ValueError("phase-9 calibration belongs to a different shortlist")
    if record["head_sha256"] != head_sha256 or record["encoder_delta_sha256"] != delta_sha256:
        raise ValueError("phase-9 calibration does not match the head or encoder delta")
    if record["template_version"] != TEMPLATE_VERSION or int(record["start_layer"]) != 26:
        raise ValueError("phase-9 calibration does not match template %s at layers 26-27" % TEMPLATE_VERSION)
    return record


def load_compatible_scorer(agent, head_path, delta_path, calibration_path, checkpoint_hash, shortlist_sha256):
    """Load the private suffix, the head, and the calibration. Mismatches raise."""
    import json
    from veya.shortlist_artifact import file_sha256

    head_sha = file_sha256(head_path)
    delta_sha = file_sha256(delta_path)
    calibration_sha = file_sha256(calibration_path)
    head_blob = torch.load(head_path, map_location="cpu", weights_only=False)
    if head_blob.get("format") != FORMAT:
        raise ValueError("hypothesis head format is not %s" % FORMAT)
    meta_path = head_path + ".json"
    with open(meta_path, encoding="utf-8") as handle:
        meta = json.load(handle)
    if meta.get("base_checkpoint_sha256") != checkpoint_hash:
        raise ValueError("hypothesis head was trained on a different checkpoint")
    if meta.get("template_version") != TEMPLATE_VERSION or meta.get("scorer_id") != "hypothesis_hce2_h0":
        raise ValueError("hypothesis head is not hypothesis_hce2_h0 / %s" % TEMPLATE_VERSION)
    if int(meta.get("start_layer")) != 26:
        raise ValueError("hypothesis head expects encoder layers starting at 26")
    delta = torch.load(delta_path, map_location="cpu", weights_only=False)
    if int(delta.get("start_layer", -1)) != 26 or len(delta.get("layers", [])) != 2:
        raise ValueError("encoder delta is not layers 26-27")
    with open(calibration_path, encoding="utf-8") as handle:
        calibration = json.load(handle)
    validate_phase9_record(calibration, checkpoint_hash, shortlist_sha256, head_sha, delta_sha)
    encoder = IsolatedAdaptedEncoder(agent.model.encoder, delta)
    head = CompatibilityHead(int(head_blob["dim"]))
    head.load_state_dict(head_blob["state_dict"])
    head.to(agent.device)
    encoder.to(agent.device)
    scorer = HypothesisScorer(
        encoder, head, agent.tok, agent.device,
        scorer_id="hypothesis_hce2_h0",
        template_version=TEMPLATE_VERSION,
        variant="H0",
        calibration_id="hypothesis_hce2_h0:%s" % calibration_sha[:16],
        temperature=float(calibration["temperature"]),
    )
    scorer.head_sha256 = head_sha
    scorer.encoder_delta_sha256 = delta_sha
    scorer.calibration_sha256 = calibration_sha
    scorer.encoder_adaptation_id = "modernbert-layers-26-27+final_norm"
    return scorer


def trainable_parameter_count(module):
    return int(sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad))


def remap_scores(scores, permutation):
    """Undo a list permutation. scores[j] belongs to the candidate now at position j."""
    restored = [None] * len(permutation)
    for position, origin in enumerate(permutation):
        restored[origin] = scores[position]
    if any(value is None for value in restored):
        raise ValueError("permutation did not cover every candidate")
    return restored


class HypothesisScorer:
    """High-N Choice path. The shared encoder is not mutated when an isolated delta is used."""

    def __init__(self, encoder, head, tokenizer, device, scorer_id, template_version=TEMPLATE_VERSION,
                 variant="H0", calibration_id="uncalibrated_t1", temperature=1.0, max_length=MAX_LENGTH):
        if float(temperature) <= 0:
            raise ValueError("hypothesis temperature must be positive")
        if calibration_id == "uncalibrated_t1" and abs(float(temperature) - 1.0) > 1e-12:
            raise ValueError("an uncalibrated hypothesis scorer cannot carry a fitted temperature")
        self.encoder = encoder
        self.head = head
        self.tokenizer = tokenizer
        self.device = device
        self.scorer_id = scorer_id
        self.template_version = template_version
        self.variant = variant
        self.calibration_id = calibration_id
        self.temperature = float(temperature)
        self.max_length = max_length
        self.head_sha256 = None
        self.encoder_delta_sha256 = None
        self.calibration_sha256 = None
        self.encoder_adaptation_id = None

    def logits_for(self, state, question, hypotheses):
        self.encoder.eval()
        self.head.eval()
        grouped, tokens, truncated = score_groups(
            self.encoder, self.head, self.tokenizer, [state], question, [list(hypotheses)],
            self.device, pair_batch=max(1, len(hypotheses)), max_length=self.max_length, grad_from_layer=None,
        )
        return grouped[0].detach().float().cpu(), int(tokens[0]), int(truncated)

    def public_choice(self, state, instructions, criteria, bundle):
        from veya.release_bundle import normalize_criteria

        items = normalize_criteria(criteria)
        keys = [key for key, _value in items]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate candidate ids")
        plan = bundle.plan(len(items))
        full_count = len(items)
        if plan["path"] == "shortlist":
            picked, _stages = bundle.shortlist(state, items, plan["k"])
            triggered = True
            k = len(picked)
        else:
            picked = list(keys)
            triggered = False
            k = len(picked)
        # H0 is the label id. Criterion values are data and are not copied in.
        hypotheses = texts_for_labels(picked, self.variant)
        logits, n_tokens, truncated = self.logits_for(state, instructions, hypotheses)
        probs = choice_probabilities(logits.numpy(), self.temperature)
        choice = picked[int(np.argmax(probs))]
        return {
            "decision": "answer",
            "route": "shortlist_k8_hypothesis_hce2" if triggered else "hypothesis_direct",
            "answer": choice,
            "labels": picked,
            "probabilities": {label: round(float(prob), 4) for label, prob in zip(picked, probs)},
            "full_candidate_count": full_count,
            "shortlist_triggered": triggered,
            "k": k,
            "hypothesis_scorer_id": self.scorer_id,
            "calibration_id": self.calibration_id,
            "template_version": self.template_version,
            "hypothesis_variant": self.variant,
            "temperature": self.temperature,
            "input_tokens": n_tokens,
            "premise_truncated": truncated,
            "encoder_adaptation_id": self.encoder_adaptation_id,
        }
