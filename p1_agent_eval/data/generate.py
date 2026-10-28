"""Synthetic agent-trace generator.

Produces realistic multi-turn agent traces for *multi-hop API tasks with state
tracking over 20+ turns* (the benchmark described in the resume). Each turn
carries a hidden ground-truth quality label we later use to (a) train the
per-turn classifier and (b) compute RL rewards.

Why synthetic: you don't have your employer's production telemetry, and we need
tens of thousands of labeled turns to make the classifier/curation/RL loop real.
Point the ingest worker at your actual OTel stream later and the same code path
takes over — only the data source changes.

Run:  python -m p1_agent_eval.data.generate --n 4000 --out p1_agent_eval/data/turns.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from shared.schemas import Turn

# A small world of "APIs" the agent can call while solving a task.
TOOLS = ["search_orders", "get_customer", "check_inventory", "create_ticket",
         "issue_refund", "lookup_shipment", "update_address", "escalate"]
MODELS = ["claude-sonnet", "claude-haiku", "gpt-4o-mini", "llama3-8b"]
COUNTRIES = ["US", "GB", "DE", "IN", "BR", "JP"]

# NOTE: good and bad snippets deliberately SHARE vocabulary and structure.
# Quality comes from subtle cues (hedging, repetition, ignoring errors, wrong
# IDs), not disjoint words — so the classifier faces a real ~0.9-F1 problem and
# genuinely ambiguous turns route to the LLM judge. We also inject label noise.
GOOD_AGENT_SNIPPETS = [
    "I found order {oid} for customer {cid}. It shipped {d} via {carrier}.",
    "Confirmed: {n} units of SKU {sku} are in the {wh} warehouse.",
    "Opened ticket {tid} and linked it to order {oid} for you.",
    "The refund of ${amt} on order {oid} is approved and processing now.",
    "Shipment {ship} is in transit, expected {d}. Anything else I can do?",
    "I've updated the address for customer {cid}; the change is live.",
    "Looks like order {oid} already shipped {d}, so a refund isn't needed yet.",
    "Checked inventory for SKU {sku}: {n} in stock, reserving one for you.",
]
BAD_AGENT_SNIPPETS = [
    "I found order {oid} for customer {cid}. It shipped {d}, I think, maybe.",
    "Confirmed: {n} units of SKU {sku} — actually the lookup errored, ignoring it.",
    "Opened ticket {tid}. Wait, what was your order number again?",
    "The refund of ${amt} on order {oid} is processing. Let me refund it once more.",
    "Shipment {ship} is in transit via {carrier}... or delivered, not sure which.",
    "I've updated the address for customer {cid}, but the call returned an error.",
    "You already gave me order {oid}, but can you tell me your order number?",
    "Escalating order {oid} to a human, then escalating again, then again.",
]

USER_SNIPPETS = [
    "Where is my order {oid}?",
    "I want a refund for {oid}, it arrived broken.",
    "Can you check if SKU {sku} is in stock?",
    "Update my shipping address to {addr}.",
    "This is the third time I'm asking about {oid}.",
    "Also, what's the status of shipment {ship}?",
]


def _rand_task(rng: random.Random) -> dict:
    return dict(
        oid=f"O{rng.randint(10000, 99999)}",
        cid=f"C{rng.randint(1000, 9999)}",
        tid=f"T{rng.randint(100, 999)}",
        sku=f"SKU{rng.randint(100, 999)}",
        ship=f"S{rng.randint(10000, 99999)}",
        amt=rng.choice([19.99, 49.5, 120.0, 7.25, 299.0]),
        n=rng.randint(0, 500),
        wh=rng.choice(["NJ", "CA", "TX"]),
        carrier=rng.choice(["UPS", "FedEx", "DHL"]),
        d=f"2026-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}",
        addr=f"{rng.randint(1,999)} Main St",
    )


def gen_trace(rng: random.Random) -> list[Turn]:
    """Generate one task = one trace of 4..24 turns."""
    task = _rand_task(rng)
    n_turns = rng.randint(4, 24)
    model = rng.choice(MODELS)
    # Weaker/cheaper models produce more bad turns -> realistic signal.
    bad_rate = {"claude-sonnet": 0.10, "gpt-4o-mini": 0.18,
                "claude-haiku": 0.22, "llama3-8b": 0.30}[model]
    country = rng.choice(COUNTRIES)
    trace_id = f"trace_{rng.getrandbits(48):012x}"

    turns: list[Turn] = []
    state_ok = True  # does the agent remember earlier state?
    for i in range(n_turns):
        is_bad = rng.random() < bad_rate or (not state_ok and rng.random() < 0.4)
        tool = rng.choice(TOOLS)
        tool_ok = not (is_bad and rng.random() < 0.7)
        user_text = rng.choice(USER_SNIPPETS).format(**task)
        if is_bad:
            agent_text = rng.choice(BAD_AGENT_SNIPPETS).format(**task)
            state_ok = state_ok and rng.random() < 0.5
            label = "bad"
        else:
            agent_text = rng.choice(GOOD_AGENT_SNIPPETS).format(**task)
            label = "good"

        # Low-confidence "gray zone" turns: short/ambiguous agent replies.
        gray = rng.random() < 0.08
        if gray:
            agent_text = rng.choice(["Okay.", "Let me see.", "One moment.",
                                     "Got it, checking now."])
            label = "needs_review"

        lat = int(rng.gauss(900 if model == "claude-sonnet" else 500, 200))
        lat = max(60, lat)
        pt = rng.randint(200, 2000)
        ct = rng.randint(20, 400)
        # rough public per-token pricing, scaled to cents
        price = {"claude-sonnet": 3e-6, "claude-haiku": 8e-7,
                 "gpt-4o-mini": 6e-7, "llama3-8b": 0.0}[model]
        cost = (pt + ct) * price

        # Inject ~11% label noise to mimic imperfect human/LLM labeling. This
        # caps the achievable F1 realistically (~0.9) and is what makes the
        # low-confidence routing to the LLM judge meaningful.
        if rng.random() < 0.11:
            label = rng.choice([l for l in ("good", "bad", "needs_review") if l != label])

        # reward: good=+1, bad=-1, needs_review=0, minus small penalties
        reward = {"good": 1.0, "bad": -1.0, "needs_review": 0.0}[label]
        if not tool_ok:
            reward -= 0.5

        turns.append(Turn(
            trace_id=trace_id, turn_index=i, user_text=user_text,
            agent_text=agent_text, tool_name=tool,
            tool_args={"order_id": task["oid"]} if "order" in tool else {},
            tool_ok=tool_ok, model=model,
            prompt_tokens=pt, completion_tokens=ct, latency_ms=lat,
            cost_usd=round(cost, 6),
            label=label, label_source="ground_truth", confidence=1.0,
            reward=reward,
            tags={"country": country, "tenant": f"tenant_{rng.randint(1,20)}",
                  "prompt_version": f"v{rng.randint(1,4)}",
                  "task_type": rng.choice(["refund", "track", "inventory", "address"])},
        ))
    return turns


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000, help="approx number of turns")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="p1_agent_eval/data/turns.jsonl")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with out.open("w") as f:
        while written < args.n:
            for t in gen_trace(rng):
                f.write(json.dumps(t.__dict__) + "\n")
                written += 1

    # quick label histogram so you can sanity-check class balance
    from collections import Counter
    labels = Counter()
    for line in out.read_text().splitlines():
        labels[json.loads(line)["label"]] += 1
    print(f"wrote {written} turns -> {out}")
    print("label distribution:", dict(labels))


if __name__ == "__main__":
    main()
