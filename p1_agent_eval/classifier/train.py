"""Train the per-turn classifier and report F1.

  python -m p1_agent_eval.classifier.train                 # auto backend
  python -m p1_agent_eval.classifier.train --backend modernbert --onnx
  python -m p1_agent_eval.classifier.train --backend tfidf

Reads turns.jsonl (from data.generate), trains, evaluates on a held-out split,
prints per-class + macro F1, and saves artifacts the inference service loads.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split

from .featurize import turn_to_text, LABELS
from .model import make_backend, ART_DIR, TfidfBackend


def load_turns(path: str):
    texts, labels = [], []
    for line in Path(path).read_text().splitlines():
        d = json.loads(line)
        texts.append(turn_to_text(d))
        labels.append(d["label"])
    return texts, labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="p1_agent_eval/data/turns.jsonl")
    ap.add_argument("--backend", default="auto", choices=["auto", "tfidf", "modernbert"])
    ap.add_argument("--onnx", action="store_true", help="export ONNX (modernbert only)")
    ap.add_argument("--epochs", type=int, default=2)
    args = ap.parse_args()

    texts, labels = load_turns(args.data)
    Xtr, Xte, ytr, yte = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels)
    print(f"train={len(Xtr)} test={len(Xte)}")

    backend = make_backend(args.backend)
    print(f"backend = {backend.name}")
    if backend.name == "modernbert":
        backend.fit(Xtr, ytr, epochs=args.epochs)
    else:
        backend.fit(Xtr, ytr)

    pred, conf = backend.predict(Xte)
    macro = f1_score(yte, pred, average="macro", labels=LABELS, zero_division=0)
    weighted = f1_score(yte, pred, average="weighted", labels=LABELS, zero_division=0)
    print("\n" + classification_report(yte, pred, labels=LABELS, zero_division=0))
    print(f"macro-F1={macro:.3f}  weighted-F1={weighted:.3f}")
    print(f"mean confidence={sum(conf)/len(conf):.3f}")

    # save artifacts
    if backend.name == "tfidf":
        p = ART_DIR / "tfidf.pkl"
        backend.save(p)
        print(f"saved -> {p}")
    else:
        backend.save(ART_DIR / "modernbert")
        print(f"saved -> {ART_DIR/'modernbert'}")
        if args.onnx:
            backend.export_onnx(ART_DIR / "classifier.onnx")

    # persist the test metrics for the dashboard
    (ART_DIR / "metrics.json").write_text(json.dumps(
        {"backend": backend.name, "macro_f1": round(macro, 4),
         "weighted_f1": round(weighted, 4), "n_test": len(Xte)}, indent=2))


if __name__ == "__main__":
    main()
