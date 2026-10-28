"""LLM-as-judge — the slow, expensive, accurate fallback.

The whole point of the ModernBERT classifier is to AVOID calling this on every
turn. We only call it on low-confidence turns (the nuanced multi-turn cases the
cheap classifier misses). This module provides the judge with graceful
degradation across backends so it always works:

    Claude API  ->  OpenAI API  ->  local Ollama  ->  deterministic heuristic

The heuristic path means the offline demo and tests run with zero API keys and
still exercise the routing logic.
"""
from __future__ import annotations

import json
import re

from shared import config
from .featurize import LABELS

SYSTEM = (
    "You are grading a single turn of a customer-support agent. "
    "Reply with ONLY one word: good, bad, or needs_review. "
    "good = correct, helpful, uses tools properly. "
    "bad = wrong, hallucinated, ignores errors, loops, or re-asks known info. "
    "needs_review = genuinely ambiguous."
)

_BAD_CUES = ["i think", "maybe", "not sure", "once more", "again, then",
             "ignoring", "errored", "returned an error", "what was your",
             "tell me your order", "or delivered"]
_GOOD_CUES = ["confirmed", "approved and processing", "is live", "in transit, expected",
              "linked it to", "reserving one"]


def _heuristic(text: str) -> tuple[str, str]:
    t = text.lower()
    bad = sum(c in t for c in _BAD_CUES)
    good = sum(c in t for c in _GOOD_CUES)
    if "[tool:" in t and "=error" in t:
        bad += 1
    if bad > good:
        return "bad", "heuristic_judge"
    if good > bad:
        return "good", "heuristic_judge"
    return "needs_review", "heuristic_judge"


def _parse(word: str) -> str | None:
    m = re.search(r"good|bad|needs_review", word.lower())
    return m.group(0) if m else None


def judge_turn(text: str) -> tuple[str, str]:
    """Return (label, source). Tries the best available backend, falls back."""
    # 1) Claude
    if config.ANTHROPIC_API_KEY and not config.USE_LOCAL_LLM:
        try:
            import anthropic
            c = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
            msg = c.messages.create(
                model=config.ANTHROPIC_MODEL, max_tokens=5,
                system=SYSTEM, messages=[{"role": "user", "content": text}])
            lab = _parse(msg.content[0].text)
            if lab:
                return lab, "claude_judge"
        except Exception:
            pass
    # 2) OpenAI
    if config.OPENAI_API_KEY and not config.USE_LOCAL_LLM:
        try:
            from openai import OpenAI
            o = OpenAI(api_key=config.OPENAI_API_KEY)
            r = o.chat.completions.create(
                model="gpt-4o-mini", max_tokens=5,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": text}])
            lab = _parse(r.choices[0].message.content)
            if lab:
                return lab, "openai_judge"
        except Exception:
            pass
    # 3) local Ollama
    if config.USE_LOCAL_LLM:
        try:
            import httpx
            resp = httpx.post(f"{config.OLLAMA_URL}/api/generate", timeout=20, json={
                "model": "llama3.2:3b", "stream": False,
                "prompt": f"{SYSTEM}\n\nTurn:\n{text}\n\nAnswer:"})
            lab = _parse(resp.json().get("response", ""))
            if lab:
                return lab, "ollama_judge"
        except Exception:
            pass
    # 4) deterministic heuristic (always works)
    return _heuristic(text)


if __name__ == "__main__":
    for s in ["[agent] The refund is approved and processing now. [tool:issue_refund=ok]",
              "[agent] I think it shipped, maybe. [tool:lookup_shipment=ERROR]",
              "[agent] One moment."]:
        print(judge_turn(s), "  <-", s[:60])
