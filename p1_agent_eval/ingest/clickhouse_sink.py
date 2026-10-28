"""ClickHouse sink. Thin wrapper so the worker doesn't care about the client."""
from __future__ import annotations

from shared import config
from shared.schemas import Turn


class ClickHouseSink:
    def __init__(self, url: str | None = None, db: str | None = None):
        import clickhouse_connect
        url = url or config.CLICKHOUSE_URL
        host = url.split("://")[-1].split(":")[0]
        port = int(url.rsplit(":", 1)[-1]) if ":" in url.split("://")[-1] else 8123
        self.client = clickhouse_connect.get_client(
            host=host, port=port, database=db or config.CLICKHOUSE_DB)
        self.cols = ["ts", "trace_id", "turn_id", "turn_index", "user_text",
                     "agent_text", "tool_name", "tool_args", "tool_ok", "model",
                     "prompt_tokens", "completion_tokens", "latency_ms", "cost_usd",
                     "label", "label_source", "confidence", "reward", "tags"]

    def write(self, turns: list[Turn]) -> int:
        import json as _json
        from datetime import datetime
        rows = []
        for t in turns:
            rows.append([
                datetime.fromtimestamp(t.ts), t.trace_id, t.turn_id, t.turn_index,
                t.user_text, t.agent_text, t.tool_name, _json.dumps(t.tool_args),
                int(t.tool_ok), t.model, t.prompt_tokens, t.completion_tokens,
                t.latency_ms, t.cost_usd, t.label, t.label_source,
                float(t.confidence), float(t.reward), t.tags,
            ])
        self.client.insert("turns", rows, column_names=self.cols)
        return len(rows)


class StdoutSink:
    """No-infra sink used for local smoke tests."""
    def __init__(self, *_, **__):
        self.count = 0

    def write(self, turns: list[Turn]) -> int:
        for t in turns:
            print(f"  [{t.label:12}|{t.label_source:14}|{t.confidence:.2f}] "
                  f"{t.agent_text[:60]}")
        self.count += len(turns)
        return len(turns)
