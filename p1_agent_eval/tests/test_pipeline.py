"""Fast tests for Project 1. No infra, no network. Run: .venv/bin/pytest p1_agent_eval"""
import json
import random
from pathlib import Path

from p1_agent_eval.data.generate import gen_trace
from p1_agent_eval.classifier.featurize import turn_to_text, LABELS
from p1_agent_eval.classifier.model import TfidfBackend
from p1_agent_eval.classifier.judge import judge_turn
from p1_agent_eval.curation.curate import build_eval_set, build_dpo_pairs, build_golden_paths


def _turns(n_traces=60):
    rng = random.Random(0)
    turns = []
    for _ in range(n_traces):
        for t in gen_trace(rng):
            turns.append(t.__dict__)
    return turns


def test_generate_shapes():
    turns = _turns()
    assert len(turns) > 100
    for t in turns[:20]:
        assert t["label"] in LABELS
        assert t["trace_id"].startswith("trace_")
        assert isinstance(t["tags"], dict)


def test_classifier_learns_signal():
    turns = _turns(120)
    texts = [turn_to_text(t) for t in turns]
    labels = [t["label"] for t in turns]
    split = int(len(texts) * 0.8)
    b = TfidfBackend()
    b.fit(texts[:split], labels[:split])
    preds, confs = b.predict(texts[split:])
    acc = sum(p == y for p, y in zip(preds, labels[split:])) / len(preds)
    assert acc > 0.6                      # well above 3-class chance (0.33)
    assert all(0.0 <= c <= 1.0 for c in confs)


def test_judge_heuristic_offline():
    # no API keys -> heuristic path, must still return a valid label
    lab, src = judge_turn("[agent] approved and processing [tool:issue_refund=ok]")
    assert lab in LABELS and "judge" in src


def test_curation_outputs_nonempty():
    turns = _turns(200)
    assert len(build_eval_set(turns)) > 0
    pairs = build_dpo_pairs(turns, max_pairs=500)
    assert len(pairs) > 0
    for p in pairs[:5]:
        assert set(p) == {"prompt", "chosen", "rejected"}
    # golden paths may be rare but structure must be valid when present
    for g in build_golden_paths(turns)[:3]:
        assert g["messages"] and g["messages"][0]["role"] == "user"
