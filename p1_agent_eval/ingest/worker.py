"""Ingest worker: raw turns -> classifier -> ClickHouse.

This is the "triage-to-dataset in <2h" pipeline. It:
  1. reads raw turns (from Kafka in prod, or the OTel span file / a jsonl replay),
  2. scores each turn with the ModernBERT classifier (+ low-conf LLM routing),
  3. writes the scored, labeled turn to ClickHouse continuously.

Sources:   kafka  | file (OTel collector's spans.jsonl) | replay (a jsonl)
Sinks:     clickhouse | stdout

Examples:
  # fully local smoke test, no infra:
  python -m p1_agent_eval.ingest.worker --source replay \
      --path p1_agent_eval/data/turns.jsonl --sink stdout --limit 20

  # production-ish: consume Kafka, write ClickHouse:
  python -m p1_agent_eval.ingest.worker --source kafka --sink clickhouse
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from shared import config
from shared.schemas import Turn
from ..classifier.infer import TurnScorer
from .clickhouse_sink import ClickHouseSink, StdoutSink


def span_to_turn(span: dict) -> Turn | None:
    """Map an OTel span (file exporter JSON) to a Turn. Agents set span
    attributes like agent.user_text, agent.tool_name, etc. (see otel_agent.py)."""
    attrs = span.get("attributes", {})
    if "agent.trace_id" not in attrs and "trace_id" not in attrs:
        return None
    return Turn(
        trace_id=attrs.get("agent.trace_id") or attrs.get("trace_id", "unknown"),
        turn_index=int(attrs.get("agent.turn_index", 0)),
        user_text=attrs.get("agent.user_text", ""),
        agent_text=attrs.get("agent.agent_text", ""),
        tool_name=attrs.get("agent.tool_name", ""),
        tool_ok=bool(attrs.get("agent.tool_ok", True)),
        model=attrs.get("agent.model", "unknown"),
        latency_ms=int(attrs.get("agent.latency_ms", 0)),
        cost_usd=float(attrs.get("agent.cost_usd", 0.0)),
        tags={k.split(".", 1)[-1]: str(v) for k, v in attrs.items()
              if k.startswith("tag.")},
    )


def iter_source(args):
    """Yield raw Turn objects from the chosen source."""
    if args.source == "replay":
        lines = Path(args.path).read_text().splitlines()
        if args.limit:
            lines = lines[: args.limit]
        for line in lines:
            d = json.loads(line)
            # strip stored ground-truth label so the classifier assigns a fresh one
            d["label"] = d["label_source"] = ""
            yield Turn(**{k: v for k, v in d.items() if k in Turn.__annotations__})

    elif args.source == "file":
        # tail the OTel collector's span file
        p = Path(args.path)
        p.touch(exist_ok=True)
        with p.open() as f:
            f.seek(0, 2)
            while True:
                line = f.readline()
                if not line:
                    time.sleep(0.5); continue
                try:
                    for span in json.loads(line).get("resourceSpans", []):
                        for ss in span.get("scopeSpans", []):
                            for sp in ss.get("spans", []):
                                t = span_to_turn(sp)
                                if t:
                                    yield t
                except Exception:
                    continue

    elif args.source == "kafka":
        from confluent_kafka import Consumer
        c = Consumer({"bootstrap.servers": config.KAFKA_BROKERS,
                      "group.id": "ingest-worker", "auto.offset.reset": "earliest"})
        c.subscribe([config.TOPIC_TURNS_RAW])
        try:
            while True:
                msg = c.poll(1.0)
                if msg is None or msg.error():
                    continue
                yield Turn.from_kafka(msg.value())
        finally:
            c.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="replay", choices=["replay", "file", "kafka"])
    ap.add_argument("--sink", default="stdout", choices=["stdout", "clickhouse"])
    ap.add_argument("--path", default="p1_agent_eval/data/turns.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=100)
    args = ap.parse_args()

    scorer = TurnScorer()
    sink = ClickHouseSink() if args.sink == "clickhouse" else StdoutSink()

    buf: list[Turn] = []
    n = 0
    t0 = time.time()
    for turn in iter_source(args):
        s = scorer.score(turn)
        turn.label, turn.label_source, turn.confidence = s.label, s.source, s.confidence
        turn.reward = {"good": 1.0, "needs_review": 0.0, "bad": -1.0}.get(s.label, 0.0)
        if not turn.tool_ok:
            turn.reward -= 0.5
        buf.append(turn)
        if len(buf) >= args.batch:
            sink.write(buf); n += len(buf); buf = []
    if buf:
        sink.write(buf); n += len(buf)

    dt = time.time() - t0
    print(f"\ningested {n} turns in {dt:.1f}s "
          f"({n/max(dt,1e-6):.0f}/s), escalated {scorer.escalation_rate:.1%} to LLM")


if __name__ == "__main__":
    main()
