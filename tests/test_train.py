"""Offline tests for veYa.train (RLCD fine-tuning helpers).

Everything here runs on CPU with no pretrained weights and no `datasets`
install, so it is safe in CI. A tiny fake tokenizer stands in for the real one
in `build_sequence`; the real checkpoint path is exercised by the notebook.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from veya.common import QTYPES, TEMP_MAX, TEMP_MIN  # noqa: E402
from veya.train import (  # noqa: E402
    TrainConfig,
    build_training_item,
    collate_train_batch,
    compute_rcld_loss,
    fit_temperature_soft,
    set_global_seed,
    split_items,
)


class _FakeTok:
    """Char-level stand-in exposing only what build_sequence touches."""

    mask_token = "[MASK]"
    mask_token_id = 3
    cls_token_id = 1
    sep_token_id = 2
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        ids = [(ord(c) % 40) + 5 for c in str(text)]
        return {"input_ids": ids}


def _items(n):
    out = []
    for i in range(n):
        out.append(
            {
                "ids": [1, 5, 6, 2],
                "markers": [1, 2],
                "qtype": QTYPES["choice"],
                "target": [1.0, 0.0],
                "label": 0,
            }
        )
    return out


def test_config_rejects_bad_values():
    for bad in (
        dict(epochs=0),
        dict(micro_batch=0),
        dict(group_size=1),
        dict(lr_encoder=0.0),
        dict(sigma_start=0.0),
        dict(rl_weight=0.0, ce_weight=0.0),
    ):
        try:
            TrainConfig(**bad)
            raise AssertionError("TrainConfig accepted invalid %r" % (bad,))
        except ValueError:
            pass


def test_sigma_anneals_between_bounds():
    cfg = TrainConfig(epochs=4, sigma_start=0.4, sigma_end=0.1)
    assert abs(cfg.sigma_for_epoch(0) - 0.4) < 1e-6
    assert abs(cfg.sigma_for_epoch(3) - 0.1) < 1e-6
    mid = cfg.sigma_for_epoch(1)
    assert 0.1 < mid < 0.4


def test_set_global_seed_is_reproducible():
    set_global_seed(123)
    a = torch.randn(4)
    set_global_seed(123)
    b = torch.randn(4)
    assert torch.allclose(a, b)


def test_split_items_has_no_overlap_and_is_deterministic():
    items = _items(100)
    train, calib = split_items(items, stride=15, max_calib=400)
    assert len(train) + len(calib) == 100
    assert len(calib) == len(range(0, 100, 15))
    # identity-based check: no object appears in both
    train_ids = {id(x) for x in train}
    assert all(id(x) not in train_ids for x in calib)
    train2, calib2 = split_items(items, stride=15, max_calib=400)
    assert len(calib2) == len(calib) and len(train2) == len(train)


def test_split_items_respects_max_calib():
    items = _items(1000)
    _train, calib = split_items(items, stride=3, max_calib=10)
    assert len(calib) == 10


def test_fit_temperature_soft_clamps_to_serving_range():
    # Near-uniform logits against near-one-hot targets: the unconstrained NLL
    # optimum would sharpen hard (T -> 0), which the notebook used to allow down
    # to 0.1. Serving clamps to TEMP_MIN, so the fit must return TEMP_MIN too --
    # this is the train/serve skew regression test.
    pairs = [(np.array([0.0, 0.01]), np.array([1.0, 0.0])) for _ in range(40)]
    t = fit_temperature_soft(pairs)
    assert t >= TEMP_MIN - 1e-9
    assert t <= TEMP_MAX + 1e-9


def test_fit_temperature_soft_needs_min_obs():
    assert fit_temperature_soft([(np.array([0.0, 1.0]), np.array([0.0, 1.0]))], min_obs=10) == 1.0


def test_collate_train_batch_shapes():
    items = _items(3)
    items[1]["ids"] = [1, 5, 6, 7, 8, 2]  # longer row forces padding
    batch = collate_train_batch(items, pad_id=0)
    assert batch["input_ids"].shape == (3, 6)
    assert batch["marker_mask"].shape == (3, 2)
    assert batch["target"].shape == (3, 2)
    assert int(batch["attention_mask"][0].sum()) == 4
    assert int(batch["attention_mask"][1].sum()) == 6


def test_build_training_item_normalizes_soft_target():
    tok = _FakeTok()
    q = {"type": "choice", "instructions": "pick", "criteria": {"a": "x", "b": "y"}}
    gold = {"probabilities": {"a": 2.0, "b": 2.0}}
    it = build_training_item(tok, {"body": "hello"}, q, gold, max_len=64, head_max_len=32)
    assert it is not None
    assert abs(sum(it["target"]) - 1.0) < 1e-6
    assert it["qtype"] == QTYPES["choice"]
    assert len(it["markers"]) == 2


def _loss_inputs(n=4, k=3, requires_grad=True):
    torch.manual_seed(0)
    logits = torch.randn(n, k, requires_grad=requires_grad)
    act = torch.randn(n, 2, requires_grad=requires_grad)
    target = torch.softmax(torch.randn(n, k), -1)
    qtype = torch.zeros(n, dtype=torch.long)
    mask = torch.ones(n, k, dtype=torch.bool)
    return logits, act, target, qtype, mask


def test_compute_rcld_loss_full_is_finite_and_backprops():
    logits, act, target, qtype, mask = _loss_inputs()
    loss, diag = compute_rcld_loss(
        logits, act, target, qtype, mask, sigma=0.3, group_size=4
    )
    assert loss.dim() == 0 and torch.isfinite(loss)
    assert "reward" in diag and "loss_rl" in diag and "loss_ce" in diag
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_compute_rcld_loss_ce_only_ablation():
    logits, act, target, qtype, mask = _loss_inputs()
    loss, diag = compute_rcld_loss(
        logits, act, target, qtype, mask, sigma=0.3, group_size=4, rl_weight=0.0, ce_weight=1.0
    )
    assert "reward" not in diag
    assert "loss_ce" in diag
    assert torch.isfinite(loss)


def test_compute_rcld_loss_act_weight_trains_act_head():
    logits, act, target, qtype, mask = _loss_inputs()
    loss, _ = compute_rcld_loss(
        logits, act, target, qtype, mask, sigma=0.3, group_size=4, act_weight=0.5
    )
    loss.backward()
    assert act.grad is not None
    assert float(act.grad.abs().sum()) > 0.0


def test_compute_rcld_loss_rejects_bad_sigma():
    logits, act, target, qtype, mask = _loss_inputs(requires_grad=False)
    try:
        compute_rcld_loss(logits, act, target, qtype, mask, sigma=0.0, group_size=4)
        raise AssertionError("expected ValueError for sigma=0")
    except ValueError:
        pass


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
    print("all train tests passed (%d)" % len(fns))


if __name__ == "__main__":
    _run_all()
