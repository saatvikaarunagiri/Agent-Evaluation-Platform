"""Shared data schemas used across the three platforms.

Kept dependency-light (dataclasses + stdlib) so every project can import these
without pulling in heavy libs. Pydantic versions live inside each service where
validation at the API boundary matters.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# --------------------------------------------------------------------------- #
# Project 1 — agent telemetry                                                 #
# --------------------------------------------------------------------------- #
@dataclass
class Turn:
    """One turn of an agent conversation. Matches telemetry.turns in ClickHouse."""
    trace_id: str
    turn_index: int
    user_text: str = ""
    agent_text: str = ""
    tool_name: str = ""
    tool_args: dict[str, Any] = field(default_factory=dict)
    tool_ok: bool = True
    model: str = "claude-sonnet"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    cost_usd: float = 0.0
    label: str = ""              # good | bad | needs_review | ""
    label_source: str = ""       # modernbert | llm_judge | human | ""
    confidence: float = 0.0
    reward: float = 0.0
    tags: dict[str, str] = field(default_factory=dict)
    turn_id: str = field(default_factory=lambda: _id("turn"))
    ts: float = field(default_factory=time.time)

    def to_kafka(self) -> bytes:
        return json.dumps(asdict(self)).encode()

    @staticmethod
    def from_kafka(raw: bytes) -> "Turn":
        d = json.loads(raw)
        d.pop("ts", None)          # recompute on ingest if absent
        return Turn(**{k: v for k, v in d.items() if k in Turn.__annotations__})


# --------------------------------------------------------------------------- #
# Project 2 — transactions / events                                           #
# --------------------------------------------------------------------------- #
@dataclass
class Transaction:
    """A payment event flowing through Kafka -> stream features -> models."""
    user_id: str
    amount: float
    merchant: str
    merchant_category: str
    country: str
    device_id: str
    is_fraud: int = 0            # ground-truth label (synthetic data only)
    txn_id: str = field(default_factory=lambda: _id("txn"))
    ts: float = field(default_factory=time.time)

    def to_kafka(self) -> bytes:
        return json.dumps(asdict(self)).encode()

    @staticmethod
    def from_kafka(raw: bytes) -> "Transaction":
        return Transaction(**json.loads(raw))
