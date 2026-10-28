"""End-to-end offline demo of the closed-loop agent-eval + RL platform.

Runs the ENTIRE loop with no Docker, no infra, no API keys:

  generate telemetry -> train classifier -> score turns (w/ LLM routing)
    -> curate datasets -> DPO warm start -> online GRPO (classifier reward)

Prints a single summary so you can see the whole system work in ~30s.

  python -m p1_agent_eval.demo
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

C = Console()
PY = sys.executable


def step(title: str, args: list[str]) -> None:
    C.rule(f"[bold]{title}")
    subprocess.run([PY, "-m", *args], check=True)


def main() -> None:
    C.print("[bold cyan]Closed-Loop Agent Eval + RL — end-to-end demo[/]\n")

    step("1/6  Generate synthetic telemetry", ["p1_agent_eval.data.generate", "--n", "8000"])
    step("2/6  Train per-turn classifier", ["p1_agent_eval.classifier.train", "--backend", "tfidf"])
    step("3/6  Ingest + score (low-conf -> LLM judge)",
         ["p1_agent_eval.ingest.worker", "--source", "replay", "--sink", "stdout", "--limit", "0"])
    step("4/6  Auto-curate datasets", ["p1_agent_eval.curation.curate", "--source", "file"])
    step("5/6  DPO warm start", ["p1_agent_eval.posttrain.dpo", "--dry-run"])
    step("6/6  Online GRPO (classifier = reward model)",
         ["p1_agent_eval.posttrain.grpo", "--dry-run", "--steps", "80"])

    # summarize
    C.rule("[bold green]Summary")
    clf = json.loads(Path("p1_agent_eval/classifier/artifacts/metrics.json").read_text())
    grpo = json.loads(Path("p1_agent_eval/posttrain/artifacts/grpo_metrics.json").read_text())
    curated = Path("p1_agent_eval/data/curated")
    counts = {p.name: sum(1 for _ in p.open()) for p in curated.glob("*.jsonl")}

    t = Table(show_header=False, box=None)
    t.add_row("Classifier backend", clf["backend"])
    t.add_row("Classifier weighted-F1", f"{clf['weighted_f1']:.3f}  (ModernBERT path -> higher)")
    t.add_row("Curated eval cases", str(counts.get("eval.jsonl", 0)))
    t.add_row("Curated DPO pairs", str(counts.get("dpo_pairs.jsonl", 0)))
    t.add_row("GRPO task-success lift", f"{grpo['task_success_lift_pct']}%")
    t.add_row("GRPO off-policy drift cut", f"{grpo['drift_reduction_pct']}%")
    C.print(t)
    C.print("\n[dim]Swap the tfidf backend for ModernBERT and the dry-run trainers "
            "for the real TRL ones to reproduce production numbers.[/]")


if __name__ == "__main__":
    main()
