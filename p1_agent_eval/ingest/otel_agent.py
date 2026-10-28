"""Example: an agent instrumented with OpenTelemetry.

This is how your REAL agent emits per-turn spans to the OTel collector, which
forwards them to the ingest worker. Each turn = one span with agent.* attributes
the worker knows how to read (see span_to_turn in worker.py).

Run the collector first (`make up-agent`), then:
    python -m p1_agent_eval.ingest.otel_agent
and watch turns appear in the worker running with --source file.
"""
from __future__ import annotations

import time

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

from ..data.generate import gen_trace
import random


def setup_tracer(endpoint: str = "http://localhost:4318/v1/traces"):
    provider = TracerProvider()
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)
    return trace.get_tracer("agent")


def emit_turn(tracer, turn) -> None:
    with tracer.start_as_current_span("agent.turn") as span:
        span.set_attribute("agent.trace_id", turn.trace_id)
        span.set_attribute("agent.turn_index", turn.turn_index)
        span.set_attribute("agent.user_text", turn.user_text)
        span.set_attribute("agent.agent_text", turn.agent_text)
        span.set_attribute("agent.tool_name", turn.tool_name)
        span.set_attribute("agent.tool_ok", turn.tool_ok)
        span.set_attribute("agent.model", turn.model)
        span.set_attribute("agent.latency_ms", turn.latency_ms)
        span.set_attribute("agent.cost_usd", turn.cost_usd)
        for k, v in turn.tags.items():
            span.set_attribute(f"tag.{k}", v)


def main() -> None:
    tracer = setup_tracer()
    rng = random.Random()
    print("emitting synthetic agent traces to the OTel collector... Ctrl-C to stop")
    try:
        while True:
            for turn in gen_trace(rng):
                emit_turn(tracer, turn)
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
