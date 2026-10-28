"""TurnClassifier — the per-turn quality classifier.

Two interchangeable backends behind one interface:

  * "modernbert"  — fine-tuned answerdotai/ModernBERT-base sequence classifier.
                    This is the production path from the resume (0.91 F1, <90ms
                    via ONNX). Needs torch+transformers (~1GB). Runs great on the
                    M5 Neural Engine via ONNX Runtime CoreML.

  * "tfidf"       — TF-IDF + LogisticRegression. ~2MB, trains in seconds, no
                    heavy deps. Used by the offline demo and the test suite, and
                    a perfectly good fallback if you don't want the download.

Both expose:  .fit(texts, labels) / .predict(texts) -> (labels, confidences)
Confidence = max softmax prob, which drives low-confidence routing to the LLM
judge (see infer.py).
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np

from .featurize import LABELS, LABEL2ID, ID2LABEL

ART_DIR = Path(os.getenv("P1_ART_DIR", "p1_agent_eval/classifier/artifacts"))
ART_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Backend 1: TF-IDF + LogisticRegression (fast, dependency-light)             #
# --------------------------------------------------------------------------- #
class TfidfBackend:
    name = "tfidf"

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000)
        self.clf = LogisticRegression(max_iter=1000, C=4.0, class_weight="balanced")

    def fit(self, texts, labels):
        y = np.array([LABEL2ID[l] for l in labels])
        X = self.vec.fit_transform(texts)
        self.clf.fit(X, y)

    def predict(self, texts):
        X = self.vec.transform(texts)
        proba = self.clf.predict_proba(X)
        # align columns to global LABELS order (clf.classes_ may be a subset)
        full = np.zeros((proba.shape[0], len(LABELS)))
        for j, c in enumerate(self.clf.classes_):
            full[:, c] = proba[:, j]
        idx = full.argmax(1)
        return [ID2LABEL[i] for i in idx], full.max(1).tolist()

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump({"vec": self.vec, "clf": self.clf}, f)

    @classmethod
    def load(cls, path: Path):
        obj = cls.__new__(cls)
        with open(path, "rb") as f:
            d = pickle.load(f)
        obj.vec, obj.clf = d["vec"], d["clf"]
        return obj


# --------------------------------------------------------------------------- #
# Backend 2: ModernBERT sequence classifier (the real deal)                   #
# --------------------------------------------------------------------------- #
class ModernBertBackend:
    name = "modernbert"
    MODEL_ID = "answerdotai/ModernBERT-base"

    def __init__(self, device: str | None = None):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.torch = torch
        self.device = device or ("mps" if torch.backends.mps.is_available()
                                 else "cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(self.MODEL_ID)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.MODEL_ID, num_labels=len(LABELS),
            id2label=ID2LABEL, label2id=LABEL2ID,
        ).to(self.device)

    def fit(self, texts, labels, epochs: int = 2, bs: int = 16, lr: float = 2e-5):
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        enc = self.tok(list(texts), truncation=True, padding=True,
                       max_length=256, return_tensors="pt")
        y = torch.tensor([LABEL2ID[l] for l in labels])
        ds = TensorDataset(enc["input_ids"], enc["attention_mask"], y)
        dl = DataLoader(ds, batch_size=bs, shuffle=True)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.model.train()
        for ep in range(epochs):
            tot = 0.0
            for ids, mask, yb in dl:
                ids, mask, yb = ids.to(self.device), mask.to(self.device), yb.to(self.device)
                out = self.model(input_ids=ids, attention_mask=mask, labels=yb)
                out.loss.backward(); opt.step(); opt.zero_grad()
                tot += out.loss.item()
            print(f"  epoch {ep+1}/{epochs}  loss={tot/len(dl):.4f}")

    def predict(self, texts, bs: int = 32):
        import torch
        self.model.eval()
        labels_out, conf_out = [], []
        with torch.no_grad():
            for i in range(0, len(texts), bs):
                batch = list(texts[i:i + bs])
                enc = self.tok(batch, truncation=True, padding=True,
                               max_length=256, return_tensors="pt").to(self.device)
                logits = self.model(**enc).logits
                proba = torch.softmax(logits, dim=-1).cpu().numpy()
                idx = proba.argmax(1)
                labels_out += [ID2LABEL[int(k)] for k in idx]
                conf_out += proba.max(1).tolist()
        return labels_out, conf_out

    def save(self, path: Path):
        self.model.save_pretrained(path)
        self.tok.save_pretrained(path)

    def export_onnx(self, path: Path):
        """Export to ONNX for <90ms inference via onnxruntime (CoreML on Mac)."""
        import torch
        self.model.eval()
        dummy = self.tok("export", return_tensors="pt", padding="max_length",
                         max_length=256).to(self.device)
        torch.onnx.export(
            self.model, (dummy["input_ids"], dummy["attention_mask"]), str(path),
            input_names=["input_ids", "attention_mask"], output_names=["logits"],
            dynamic_axes={"input_ids": {0: "b", 1: "s"},
                          "attention_mask": {0: "b", 1: "s"},
                          "logits": {0: "b"}},
            opset_version=17,
        )
        print(f"  exported ONNX -> {path}")


# --------------------------------------------------------------------------- #
# Factory                                                                     #
# --------------------------------------------------------------------------- #
def make_backend(kind: str = "auto"):
    """kind: 'modernbert' | 'tfidf' | 'auto'. 'auto' uses ModernBERT if torch+
    transformers are importable, else tfidf."""
    if kind == "tfidf":
        return TfidfBackend()
    if kind == "modernbert":
        return ModernBertBackend()
    try:                                   # auto
        import torch, transformers  # noqa: F401
        return ModernBertBackend()
    except Exception:
        print("[classifier] transformers/torch not available — using tfidf backend")
        return TfidfBackend()
