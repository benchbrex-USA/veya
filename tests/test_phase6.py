"""Phase-6 split guards and the promotion rule. No checkpoint is loaded."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.veyabench.phase6_protocol import (  # noqa: E402
    assert_training_id, confusion_graph, example_id, paired_bootstrap, pair_relation, promote,
)

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name if cond else "%s %s" % (name, detail))


def test_train_ids_refuse_the_release_test_and_the_val_slice():
    assert_training_id(example_id("train", 800))
    assert_training_id(example_id("train", 8599))
    check("train/800 accepted", True)
    for bad in ("test:1400", "train:100", "train:9000", "train:9500"):
        try:
            if bad.startswith("train:"):
                assert_training_id(bad)
            else:
                assert_training_id(bad)
            check("refuses %s" % bad, False)
        except ValueError:
            check("refuses %s" % bad, True)


def test_promotion_requires_a_positive_interval_and_no_extra_confident_errors():
    ok, _reason = promote(0.01, 0.10, 0.12)
    check("promote/clear gain", ok)
    ok, _reason = promote(-0.001, 0.10, 0.12)
    check("promote/interval crosses zero", ok is False)
    ok, _reason = promote(0.02, 0.20, 0.10)
    check("promote/more confident errors", ok is False)


def test_taxonomy_and_bootstrap_are_deterministic():
    relation = pair_relation("card_payment_fee_charged", "card_payment_wrong_exchange_rate")
    check("pair/same parent", relation["relation"] == "same_parent", relation["relation"])
    edges = confusion_graph([("a", "b"), ("a", "b"), ("a", "c")])
    check("graph/top edge", edges[0]["count"] == 2 and edges[0]["predicted"] == "b")
    first = paired_bootstrap([1, 0, 1, 1], [1, 1, 1, 0], n=200, seed=17)
    second = paired_bootstrap([1, 0, 1, 1], [1, 1, 1, 0], n=200, seed=17)
    check("bootstrap/stable", first == second)


def test_recorded_checks_pass():
    assert not FAIL, "\n".join(FAIL)


if __name__ == "__main__":
    test_train_ids_refuse_the_release_test_and_the_val_slice()
    test_promotion_requires_a_positive_interval_and_no_extra_confident_errors()
    test_taxonomy_and_bootstrap_are_deterministic()
    print("%d passed, %d failed" % (len(PASS), len(FAIL)))
    for item in FAIL:
        print(" FAIL", item)
    raise SystemExit(1 if FAIL else 0)
