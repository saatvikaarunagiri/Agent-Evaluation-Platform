"""Online inference with low-confidence routing.

This is what runs on every production turn:

    classifier.predict(turn)          # cheap, <90ms with ONNX ModernBERT
      └─ if confidence >= threshold -> accept the label
      └─ else                       -> route to the LLM judge (judge.py)

That routing is the cost lever from the resume: the classifier handles the
~90% confident turns for ~free, and only the hard tail hits the expensive LLM.
We track how often we escalate so you can see the cost saving.

Loads an ONNX model if present (fastest), else the saved torch/tfidf backend.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from shared import config
from .featurize import turn_to_text, ID2LABEL, LABELS
from .model import ART_DIR, TfidfBackend
from .judge import judge_turn


@dataclass
class Scored:
    label: str
    source: str          # modernbert | tfidf | *_judge
    confidence: float
    escalated: bool


class OnnxClassifier:
    """onnxruntime wrapper for the exported ModernBERT (the <90ms path)."""
    def __init__(self, onnx_path: Path):
        import onnxruntime as ort
        from transformers import AutoTokenizer
        # CoreMLExecutionProvider uses the Neural Engine on Apple Silicon.
        providers = [p for p in ("CoreMLExecutionProvider", "CPUExecutionProvider")
                     if p in ort.get_available_providers()]
        self.sess = ort.InferenceSession(str(onnx_path), providers=providers)
        self.tok = AutoTokenizer.from_pretrained(ART_DIR / "modernbert")
        self.name = "modernbert"

    def predict(self, texts):
        enc = self.tok(list(texts), truncation=True, padding="max_length",
                       max_length=256, return_tensors="np")
        logits = self.sess.run(None, {"input_ids": enc["input_ids"].astype("int64"),
                                      "attention_mask": enc["attention_mask"].astype("int64")})[0]
        e = np.exp(logits - logits.max(1, keepdims=True))
        proba = e / e.sum(1, keepdims=True)
        idx = proba.argmax(1)
        return [ID2LABEL[int(i)] for i in idx], proba.max(1).tolist()


def _load_backend():
    onnx = ART_DIR / "classifier.onnx"
    if onnx.exists():
        try:
            return OnnxClassifier(onnx)
        except Exception as e:  # noqa: BLE001
            print(f"[infer] ONNX load failed ({e}); falling back")
    mb = ART_DIR / "modernbert"
    if mb.exists():
        from .model import ModernBertBackend
        b = ModernBertBackend.__new__(ModernBertBackend)
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        b.torch = torch
        b.device = "mps" if torch.backends.mps.is_available() else "cpu"
        b.tok = AutoTokenizer.from_pretrained(mb)
        b.model = AutoModelForSequenceClassification.from_pretrained(mb).to(b.device)
        return b
    tf = ART_DIR / "tfidf.pkl"
    if tf.exists():
        return TfidfBackend.load(tf)
    raise FileNotFoundError("No classifier artifact found. Run `make p1-train-clf` first.")


class TurnScorer:
    def __init__(self, threshold: float | None = None):
        self.backend = _load_backend()
        self.threshold = config.LOW_CONF_THRESHOLD if threshold is None else threshold
        self.n_total = 0
        self.n_escalated = 0

    def score(self, turn) -> Scored:
        text = turn_to_text(turn)
        (label,), (conf,) = self.backend.predict([text])
        self.n_total += 1
        if conf >= self.threshold:
            return Scored(label, self.backend.name, conf, escalated=False)
        # low confidence -> let the LLM judge decide
        self.n_escalated += 1
        jlabel, jsource = judge_turn(text)
        return Scored(jlabel, jsource, conf, escalated=True)

    @property
    def escalation_rate(self) -> float:
        return self.n_escalated / max(1, self.n_total)


if __name__ == "__main__":
    import sys
    scorer = TurnScorer()
    data = Path("p1_agent_eval/data/turns.jsonl")
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    rows = [json.loads(l) for l in data.read_text().splitlines()[:n]]
    correct = 0
    for d in rows:
        s = scorer.score(d)
        correct += (s.label == d["label"])
    print(f"scored {len(rows)} turns")
    print(f"  accuracy vs stored labels : {correct/len(rows):.3f}")
    print(f"  escalated to LLM judge    : {scorer.n_escalated} "
          f"({scorer.escalation_rate:.1%})")
    print(f"  => online eval cost saving: ~{(1-scorer.escalation_rate):.0%} of turns "
          f"never hit the LLM")
