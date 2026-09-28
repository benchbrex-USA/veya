"""Regression tests for the K=8 packed reranker (veya/reranker.py).

Uses a tiny local checkpoint (no network, no pretrained weights) built with the
same recipe as tests/test_download.py, but with head_layers=2 so the agent has
a decision head for K8Reranker.clone_from_agent to copy.

Run: python tests/test_reranker.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path
import unittest

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
from tokenizers.models import WordLevel  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast  # noqa: E402

from veya import load  # noqa: E402
from veya.common import DecisionModel  # noqa: E402
from veya.reranker import (  # noqa: E402
    ROUTE,
    K8Reranker,
    load_reranker,
    save_reranker,
)
from veya.shortlist_artifact import file_sha256  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name if cond else "%s %s" % (name, detail))


class RerankerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.repo = Path(cls.tmp.name) / "repo"
        cls.repo.mkdir()
        config = BertConfig(vocab_size=6, hidden_size=1024, num_hidden_layers=1,
                            num_attention_heads=2, intermediate_size=4096)
        config.save_pretrained(cls.repo / "encoder")
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel(
                {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "": 4, "hello": 5},
                unk_token="[UNK]")),
            pad_token="[PAD]", unk_token="[UNK]", cls_token="[CLS]",
            sep_token="[SEP]", mask_token="",
        )
        tokenizer.save_pretrained(cls.repo / "tokenizer")
        model = DecisionModel(BertModel(config), head_layers=2)
        save_file(model.state_dict(), cls.repo / "model.safetensors")
        (cls.repo / "rl_agent_config.json").write_text(json.dumps({
            "encoder": "unused/offline", "head_layers": 2, "act_costs": {"act": 0},
            "max_len": 64, "head_max_len": 32,
        }))
        cls.agent = load(str(cls.repo), device="cpu")
        cls.criteria = {"yes": "it holds", "no": "it does not hold"}
        cls.state = "hello"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_clone_from_agent_copies_head_weights(self):
        reranker = K8Reranker()
        before = {k: v.clone() for k, v in self.agent.model.head.state_dict().items()}
        result = reranker.clone_from_agent(self.agent)
        check("clone/returns self", result is reranker)
        same = all(
            torch.equal(before[k], reranker.head.state_dict()[k]) for k in before
        )
        check("clone/head weights match agent", same)
        untouched = all(
            torch.equal(before[k], self.agent.model.head.state_dict()[k]) for k in before
        )
        check("clone/agent head stays put", untouched)
        te_same = torch.equal(
            self.agent.model.type_emb.state_dict()["weight"],
            reranker.type_emb.state_dict()["weight"],
        )
        check("clone/type_emb matches", te_same)
        sc_same = all(
            torch.equal(a, b) for a, b in zip(
                self.agent.model.scorer.state_dict().values(),
                reranker.scorer.state_dict().values(),
            )
        )
        check("clone/scorer matches", sc_same)

    def test_score_hidden_masks_padded_positions(self):
        reranker = K8Reranker().eval()
        hidden = torch.randn(2, 12, 1024)
        attention_mask = torch.ones(2, 12, dtype=torch.long)
        marker_pos = torch.tensor([[2, 5, 9], [1, 3, 4]])
        marker_mask = torch.tensor([[True, True, True], [True, True, False]])
        qtype = torch.zeros(2, dtype=torch.long)
        logits = reranker.score_hidden(hidden, attention_mask, marker_pos, marker_mask, qtype)
        check("score_hidden/shape [B,K]", tuple(logits.shape) == (2, 3), str(tuple(logits.shape)))
        check("score_hidden/float32", logits.dtype == torch.float32)
        check("score_hidden/masked slot filled", bool((logits[1, 2] == -1e4).all()))
        check("score_hidden/valid slots not fill value",
              bool((logits[0] != -1e4).all()) and bool((logits[1, :2] != -1e4).all()))

    def test_choice_logits_returns_one_logit_per_criterion(self):
        reranker = K8Reranker().clone_from_agent(self.agent)
        logits, n_tokens = reranker.choice_logits(
            self.agent, self.state, "Pick one", self.criteria,
        )
        check("choice_logits/two logits", logits.shape == (2,), str(logits.shape))
        check("choice_logits/finite", bool(torch.isfinite(torch.tensor(logits)).all()))
        check("choice_logits/token count positive", n_tokens > 0, str(n_tokens))
        check("choice_logits/numpy dtype", hasattr(logits, "dtype"), type(logits).__name__)

    def test_choice_logits_raises_when_markers_are_dropped(self):
        reranker = K8Reranker().clone_from_agent(self.agent)
        wide = {("opt%02d" % i): "x" for i in range(40)}
        try:
            reranker.choice_logits(self.agent, self.state, "Pick one", wide)
            check("choice_logits/40 options raise", False)
        except ValueError as exc:
            check("choice_logits/40 options raise", "kept" in str(exc) and "40" in str(exc), str(exc))

    def test_save_and_load_round_trip(self):
        reranker = K8Reranker().clone_from_agent(self.agent)
        path = str(Path(self.tmp.name) / "reranker.pt")
        side = save_reranker(path, reranker, checkpoint_hash="hash-1",
                             shortlist_sha256="sha-1", training_manifest={"k": 8})
        check("save/side json written", Path(path + ".json").is_file())
        check("save/side records sha256", side["file_sha256"] == file_sha256(path))
        check("save/side keeps metadata", side["training_manifest"] == {"k": 8})
        check("save/side records route", side["route"] == ROUTE and side["k"] == 8)
        loaded = load_reranker(path, self.agent, checkpoint_hash="hash-1",
                               shortlist_sha256="sha-1")
        check("load/weights match", all(
            torch.equal(a, b) for a, b in zip(
                loaded.head.state_dict().values(), reranker.head.state_dict().values())))
        check("load/meta carries blob minus state_dict", loaded.meta.get("checkpoint_hash") == "hash-1")
        check("load/meta records file sha256", loaded.meta.get("file_sha256") == file_sha256(path))
        check("load/eval mode", not loaded.training)
        before = reranker.choice_logits(self.agent, self.state, "Pick one", self.criteria)[0]
        after = loaded.choice_logits(self.agent, self.state, "Pick one", self.criteria)[0]
        check("load/same outputs", bool((torch.tensor(before) == torch.tensor(after)).all()))

    def test_load_refuses_mismatched_artifacts(self):
        reranker = K8Reranker().clone_from_agent(self.agent)
        path = str(Path(self.tmp.name) / "reranker_ref.pt")
        save_reranker(path, reranker, checkpoint_hash="hash-1",
                      shortlist_sha256="sha-1", training_manifest={})
        cases = [
            ("wrong checkpoint hash", dict(checkpoint_hash="other", shortlist_sha256="sha-1"),
             "different checkpoint"),
            ("wrong shortlist sha", dict(checkpoint_hash="hash-1", shortlist_sha256="other"),
             "different shortlist"),
        ]
        for name, kwargs, needle in cases:
            try:
                load_reranker(path, self.agent, **kwargs)
                check("load/%s refused" % name, False)
            except ValueError as exc:
                check("load/%s refused" % name, needle in str(exc), str(exc))
        bad_format = str(Path(self.tmp.name) / "reranker_bad_format.pt")
        torch.save({"format": "not-the-format", "state_dict": {}}, bad_format)
        try:
            load_reranker(bad_format, self.agent, checkpoint_hash="h", shortlist_sha256="s")
            check("load/bad format refused", False)
        except ValueError as exc:
            check("load/bad format refused", "unsupported reranker format" in str(exc), str(exc))

    def test_file_sha256_matches_hashlib(self):
        target = self.repo / "model.safetensors"
        import hashlib
        expected = hashlib.sha256(target.read_bytes()).hexdigest()
        check("file_sha256/equals hashlib", file_sha256(str(target)) == expected)


def test_recorded_checks_pass():
    assert not FAIL, "\n".join(FAIL)


if __name__ == "__main__":
    torch.set_num_threads(1)
    RerankerTests.setUpClass()
    try:
        for name in sorted(dir(RerankerTests)):
            if name.startswith("test_"):
                getattr(RerankerTests(), name)()
                print(" ok", name)
    finally:
        RerankerTests.tearDownClass()
    print("%d passed, %d failed" % (len(PASS), len(FAIL)))
    for item in FAIL:
        print(" FAIL", item)
    raise SystemExit(1 if FAIL else 0)
