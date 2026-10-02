"""First-class RLCD fine-tuning for Veya decision models.

This module is the single source of truth for the training loop that the Kaggle
notebook (``notebooks/veya_finetune_typed_decisions_2xT4_kaggle.ipynb``)
demonstrates. The notebook keeps only its DDP/``torchrun`` orchestration and
imports the math from here, so the reward, the calibration fit, and the
held-out split cannot drift between the two.

Design notes / fixes baked in here (they were previously inconsistent in the
notebook):

* **Calibration clamp matches serving.** :func:`fit_temperature_soft` clamps to
  ``[TEMP_MIN, TEMP_MAX]`` imported from :mod:`veya.common` -- the same range
  :func:`veya.common.clamp_temperature` enforces at load time. A temperature
  fitted outside that band would otherwise be silently re-clamped when the
  checkpoint is served, so the calibration the trainer reports is the
  calibration the runtime actually applies.
* **Calibration is held out from training.** :func:`split_items` removes the
  calibration rows from the training set instead of fitting on a slice of data
  the model was just trained on.
* **Reproducibility.** :func:`set_global_seed` seeds ``random``, ``numpy`` and
  ``torch`` (CPU + CUDA), so the exploration noise in :func:`compute_rcld_loss`
  is reproducible.
* **Ablation + act head are explicit.** ``rl_weight`` / ``ce_weight`` make the
  "CE-only vs CE+PG" ablation a config flag, and ``act_weight`` decides whether
  the defer/escalate head receives gradient (the notebook historically passed
  ``0.0`` -- kept here as the default so behaviour is unchanged unless asked).

The pure helpers (config validation, item building, collation, temperature fit,
loss) are CPU- and offline-safe and are covered by ``tests/test_train.py``.
``run_finetune`` and ``load_typed_decisions_items`` additionally need a
checkpoint on disk and the ``datasets`` extra (``pip install 'veya[train]'``).
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .common import (
    QTYPES,
    TEMP_MAX,
    TEMP_MIN,
    build_model,
    build_sequence,
    clamp_temperature,
    proper_reward,
    render_options,
)

__all__ = [
    "TrainConfig",
    "set_global_seed",
    "build_training_item",
    "collate_train_batch",
    "split_items",
    "fit_temperature_soft",
    "compute_rcld_loss",
    "load_typed_decisions_items",
    "run_finetune",
]


@dataclass
class TrainConfig:
    """Hyper-parameters for an RLCD fine-tune.

    Defaults reproduce the Kaggle 2xT4 recipe. ``rl_weight``/``ce_weight``
    expose the ablation; ``act_weight`` controls whether the defer/escalate head
    is trained.
    """

    epochs: int = 4
    micro_batch: int = 8
    grad_accum: int = 4
    group_size: int = 4
    lr_encoder: float = 2.5e-5
    lr_head: float = 1.0e-4
    weight_decay: float = 0.01
    sigma_start: float = 0.4
    sigma_end: float = 0.1
    w_sph: float = 0.75
    w_rps: float = 1.0
    rl_weight: float = 1.0
    ce_weight: float = 1.0
    act_weight: float = 0.0
    grad_clip: float = 1.0
    seed: int = 42
    calib_stride: int = 15
    calib_max: int = 400
    max_len: int = 1024
    head_max_len: int = 256
    model_name: str = "veya-typed-decisions"
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.micro_batch < 1 or self.grad_accum < 1:
            raise ValueError("micro_batch and grad_accum must be >= 1")
        if self.group_size < 2:
            raise ValueError("group_size must be >= 2 (GRPO needs a baseline group)")
        if self.lr_encoder <= 0 or self.lr_head <= 0:
            raise ValueError("learning rates must be positive")
        if self.sigma_start <= 0 or self.sigma_end <= 0:
            raise ValueError("sigma_start and sigma_end must be positive")
        if self.ce_weight == 0.0 and self.rl_weight == 0.0:
            raise ValueError("at least one of rl_weight / ce_weight must be non-zero")
        if self.calib_stride < 1:
            raise ValueError("calib_stride must be >= 1")

    def sigma_for_epoch(self, epoch: int) -> float:
        """Linearly anneal exploration noise across epochs (0-indexed)."""
        progress = epoch / max(1, self.epochs - 1)
        return self.sigma_start + (self.sigma_end - self.sigma_start) * progress


def set_global_seed(seed: int) -> None:
    """Seed every RNG the trainer touches, so runs are reproducible."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_training_item(
    tok: Any,
    state: Any,
    q: Dict[str, Any],
    gold_q: Dict[str, Any],
    max_len: int,
    head_max_len: int,
) -> Optional[Dict[str, Any]]:
    """Turn one (state, question, teacher-answer) triple into a training item.

    The teacher's ``probabilities`` become a **soft target** distribution over
    the rendered options. Returns ``None`` when the packed sequence could not
    keep a marker for every option (the head budget was too small), so the
    caller can drop it rather than train on a mis-aligned row.
    """
    t = q["type"]
    crit = q.get("criteria", {})
    probs = gold_q.get("probabilities", {}) if isinstance(gold_q, dict) else {}
    if t == "choice":
        keys = list(crit.keys())
        target = [float(probs.get(k, 0.0)) for k in keys]
    elif t == "noul":
        target = [float(probs.get("false", 0.5)), float(probs.get("true", 0.5))]
    elif t == "score":
        n_levels = len(crit) if isinstance(crit, list) else 4
        target = [float(probs.get(str(i), 0.0)) for i in range(n_levels)]
    else:
        raise ValueError("unsupported question type %r" % (t,))

    total = sum(target)
    target = [v / total for v in target] if total > 0 else [1.0 / len(target)] * len(target)
    label = int(np.argmax(target))
    k = len(render_options({"t": t, "crit": crit}))

    seq, markers = build_sequence(
        tok, state, {"t": t, "ins": q["instructions"], "crit": crit}, max_len, head_max_len
    )
    if len(markers) != k:
        return None
    return {
        "ids": seq,
        "markers": markers,
        "qtype": QTYPES[t],
        "target": target,
        "label": label,
    }


def collate_train_batch(items: Sequence[Dict[str, Any]], pad_id: int) -> Dict[str, torch.Tensor]:
    """Pad a list of training items into rectangular tensors."""
    if not items:
        raise ValueError("cannot collate an empty batch")
    n = len(items)
    L = max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, : len(it["target"])] = torch.tensor(it["target"], dtype=torch.float32)
    return {
        "input_ids": ids,
        "attention_mask": att,
        "marker_pos": mpos,
        "marker_mask": mmask,
        "target": target,
        "qtype": torch.tensor([it["qtype"] for it in items]),
        "label": torch.tensor([it["label"] for it in items]),
    }


def split_items(
    items: Sequence[Dict[str, Any]],
    stride: int = 15,
    max_calib: int = 400,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split into ``(train, calibration)`` with **no overlap**.

    Every ``stride``-th item (up to ``max_calib``) becomes held-out calibration;
    the rest train. Fitting temperature on rows the model never trained on is
    what makes the reported ECE honest.
    """
    if stride < 1:
        raise ValueError("stride must be >= 1")
    calib_idx = set(range(0, len(items), stride))
    if max_calib is not None and max_calib >= 0:
        keep = set(sorted(calib_idx)[:max_calib])
    else:
        keep = calib_idx
    train = [it for i, it in enumerate(items) if i not in keep]
    calib = [it for i, it in enumerate(items) if i in keep]
    return train, calib


def fit_temperature_soft(
    pairs: Sequence[Tuple[Any, Any]],
    min_obs: int = 10,
) -> float:
    """NLL-minimising temperature against **soft** teacher targets.

    ``pairs`` is a list of ``(logits[:k], target[:k])`` rows for one question
    type. The optimiser is unconstrained in log-space, but the returned value is
    clamped to ``[TEMP_MIN, TEMP_MAX]`` so it is identical to what
    :func:`veya.common.clamp_temperature` will apply at serving time.
    """
    if len(pairs) < min_obs:
        return 1.0
    kmax = max(len(z) for z, _ in pairs)
    Z = torch.full((len(pairs), kmax), -1e4)
    T = torch.zeros((len(pairs), kmax))
    for i, (z, t) in enumerate(pairs):
        z = np.asarray(z, dtype=np.float32)
        t = np.asarray(t, dtype=np.float32)
        Z[i, : len(z)] = torch.from_numpy(z)
        T[i, : len(t)] = torch.from_numpy(t)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return clamp_temperature(float(log_t.exp().item()))


def compute_rcld_loss(
    logits: torch.Tensor,
    act: torch.Tensor,
    target: torch.Tensor,
    qtype: torch.Tensor,
    mask: torch.Tensor,
    *,
    sigma: float,
    group_size: int,
    w_sph: float = 0.75,
    w_rps: float = 1.0,
    rl_weight: float = 1.0,
    ce_weight: float = 1.0,
    act_weight: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """One RLCD objective: GRPO-style policy gradient + soft cross-entropy.

    Returns the (un-scaled) loss and a diagnostics dict. ``rl_weight=0`` gives
    the pure-distillation ablation; ``act_weight>0`` trains the defer/escalate
    head instead of only keeping it in the autograd graph.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if group_size < 2:
        raise ValueError("group_size must be >= 2")
    logits = logits.float()
    k = mask.sum(-1, keepdim=True).float()

    diag: Dict[str, float] = {}
    loss = torch.zeros((), device=logits.device, dtype=torch.float32)

    if rl_weight != 0.0:
        # Sample G noisy logit distributions with a zero-mean projection so the
        # perturbation stays on the simplex tangent.
        eps = torch.randn((group_size,) + logits.shape, device=logits.device) * sigma * mask
        eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
        z = logits.detach().unsqueeze(0) + eps
        q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
        with torch.no_grad():
            r = proper_reward(q, target.unsqueeze(0), qtype, mask, w_sph=w_sph, w_rps=w_rps)
            adv = r - r.mean(0, keepdim=True)
            adv = adv / (adv.std() + 1e-6)
        logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
        loss_rl = -(adv * logp).mean()
        loss = loss + rl_weight * loss_rl
        diag["loss_rl"] = float(loss_rl.detach().cpu())
        diag["reward"] = float(r.mean().cpu())

    if ce_weight != 0.0:
        loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
        loss = loss + ce_weight * loss_ce
        diag["loss_ce"] = float(loss_ce.detach().cpu())

    # Keep the act head in the graph (DDP find_unused_parameters) and, when
    # asked, give it a real gradient.
    loss = loss + act_weight * act.sum()
    diag["act_weight"] = float(act_weight)
    return loss, diag


def load_typed_decisions_items(
    model_dir: str,
    split: str = "train",
    max_len: int = 1024,
    head_max_len: int = 256,
    dataset_name: str = "LocalLLaMA/typed-decisions",
    config_name: str = "all",
) -> List[Dict[str, Any]]:
    """Build training items from the HF ``typed-decisions`` dataset.

    Needs the ``datasets`` extra. Kept separate from the pure helpers so
    ``import veya.train`` never requires it.
    """
    from datasets import load_dataset  # lazy: only the HF-data path needs it
    from transformers import AutoTokenizer

    from .agent import _fix_tokenizer_config

    _fix_tokenizer_config(model_dir)
    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    ds = load_dataset(dataset_name, config_name, split=split)
    items: List[Dict[str, Any]] = []
    for row in ds:
        state = json.loads(row["state"])
        questions = json.loads(row["questions"])
        gold = json.loads(row["gold"])
        for qid, q in questions.items():
            if qid in gold:
                it = build_training_item(tok, state, q, gold[qid], max_len, head_max_len)
                if it:
                    items.append(it)
    return items


def run_finetune(
    model_dir: str,
    output_dir: str,
    items: Sequence[Dict[str, Any]],
    cfg: Optional[TrainConfig] = None,
    device: Optional[str] = None,
    log_every: int = 50,
    progress: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Single-process RLCD fine-tune: train, fit held-out calibration, save.

    This is the tested reference path (CPU / MPS / one GPU). For multi-GPU, wrap
    the same helpers in ``torchrun`` as the Kaggle notebook does -- the math is
    identical because both call :func:`compute_rcld_loss` and
    :func:`fit_temperature_soft`.
    """
    cfg = cfg or TrainConfig()
    emit = progress or (lambda _msg: None)
    set_global_seed(cfg.seed)

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    dev = torch.device(device)

    with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
        model_cfg = json.load(f)
    model_cfg["max_len"] = cfg.max_len
    model_cfg["head_max_len"] = cfg.head_max_len

    from transformers import AutoTokenizer

    from .agent import _fix_tokenizer_config

    _fix_tokenizer_config(model_dir)
    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    model = build_model(model_cfg, encoder_dir=os.path.join(model_dir, "encoder"))
    from safetensors.torch import load_file

    model.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), strict=True)
    model.to(dev)
    model.train()

    train_items, calib_items = split_items(items, cfg.calib_stride, cfg.calib_max)
    emit("train=%d calib=%d (held out)" % (len(train_items), len(calib_items)))

    enc_params = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    optimizer = torch.optim.AdamW(
        [
            {"params": enc_params, "lr": cfg.lr_encoder},
            {"params": head_params, "lr": cfg.lr_head},
        ],
        weight_decay=cfg.weight_decay,
    )
    updates_per_epoch = max(1, len(train_items) // (cfg.micro_batch * cfg.grad_accum))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, updates_per_epoch * cfg.epochs), eta_min=1e-6
    )
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    work = list(train_items)
    step = 0
    for epoch in range(cfg.epochs):
        rng = random.Random(cfg.seed + epoch)
        rng.shuffle(work)
        sigma = cfg.sigma_for_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        accum = 0
        running = 0.0
        for start in range(0, len(work), cfg.micro_batch):
            chunk = work[start : start + cfg.micro_batch]
            if not chunk:
                continue
            batch = collate_train_batch(chunk, tok.pad_token_id)
            batch = {k: v.to(dev) for k, v in batch.items()}
            with torch.autocast(dev.type, dtype=torch.float16, enabled=use_amp):
                logits, act = model(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["marker_pos"],
                    batch["marker_mask"],
                    batch["qtype"],
                )
            loss, diag = compute_rcld_loss(
                logits,
                act,
                batch["target"],
                batch["qtype"],
                batch["marker_mask"],
                sigma=sigma,
                group_size=cfg.group_size,
                w_sph=cfg.w_sph,
                w_rps=cfg.w_rps,
                rl_weight=cfg.rl_weight,
                ce_weight=cfg.ce_weight,
                act_weight=cfg.act_weight,
            )
            scaled = loss / cfg.grad_accum
            scaler.scale(scaled).backward()
            accum += 1
            running += float(loss.detach().cpu())
            step += 1
            if accum % cfg.grad_accum == 0 or start + cfg.micro_batch >= len(work):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if log_every and step % log_every == 0:
                emit("epoch %d step %d loss %.4f reward %.3f" % (epoch + 1, step, running / max(1, accum), diag.get("reward", 0.0)))
        emit("=== epoch %d/%d done ===" % (epoch + 1, cfg.epochs))

    temps = _fit_calibration(model, tok, calib_items, dev, cfg)
    _save(model, tok, model_cfg, temps, output_dir, cfg)
    emit("saved fine-tuned checkpoint to %s" % output_dir)
    return {"temperatures": temps, "train_items": len(train_items), "calib_items": len(calib_items)}


def _fit_calibration(model, tok, calib_items, dev, cfg: TrainConfig) -> List[float]:
    model.eval()
    preds: List[Tuple[int, np.ndarray, Any]] = []
    with torch.no_grad():
        for start in range(0, len(calib_items), 16):
            chunk = calib_items[start : start + 16]
            if not chunk:
                continue
            batch = collate_train_batch(chunk, tok.pad_token_id)
            batch = {k: v.to(dev) for k, v in batch.items()}
            logits, _ = model(
                batch["input_ids"],
                batch["attention_mask"],
                batch["marker_pos"],
                batch["marker_mask"],
                batch["qtype"],
            )
            l_np = logits.float().cpu().numpy()
            for r, it in enumerate(chunk):
                k = len(it["markers"])
                preds.append((int(it["qtype"]), l_np[r, :k], it["target"]))
    temps = [1.2, 1.2, 1.2]
    for qt in range(3):
        sel = [(z, t) for q, z, t in preds if q == qt]
        if sel:
            temps[qt] = fit_temperature_soft(sel)
    return temps


def _save(model, tok, model_cfg, temps, output_dir, cfg: TrainConfig) -> None:
    from safetensors.torch import save_file

    os.makedirs(output_dir, exist_ok=True)
    sd = {k: v.half().contiguous().cpu() for k, v in model.state_dict().items()}
    save_file(sd, os.path.join(output_dir, "model.safetensors"))
    model.encoder.config.save_pretrained(os.path.join(output_dir, "encoder"))
    tok.save_pretrained(os.path.join(output_dir, "tokenizer"))
    model_cfg["fine_tuned"] = True
    model_cfg["model_name"] = cfg.model_name
    model_cfg["temperature"] = temps
    model_cfg.setdefault("training", {})
    model_cfg["training"].update(
        {
            "epochs": cfg.epochs,
            "rl_weight": cfg.rl_weight,
            "ce_weight": cfg.ce_weight,
            "act_weight": cfg.act_weight,
            "seed": cfg.seed,
            "clamp": [TEMP_MIN, TEMP_MAX],
        }
    )
    with open(os.path.join(output_dir, "rl_agent_config.json"), "w") as f:
        json.dump(model_cfg, f, indent=2)
