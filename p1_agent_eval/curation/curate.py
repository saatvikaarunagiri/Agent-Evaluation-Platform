"""Auto-curate eval + fine-tune datasets from scored turns.

This is the payoff of streaming telemetry: instead of a human spending days
triaging logs into a dataset, we derive three artifacts automatically and
continuously:

  1. eval.jsonl        — a balanced, stratified benchmark (the "500+ case
                         benchmark"): high-agreement turns only, so labels are
                         trustworthy, stratified by task_type + model.
  2. golden_paths.jsonl — full traces where every turn is good. Used to
                         INITIALIZE post-training (DPO/SFT warm start).
  3. dpo_pairs.jsonl   — (prompt, chosen, rejected) preference pairs mined by
                         matching a good vs a bad agent response to the same
                         user intent. This is the "50K+ preference pairs".

Source: ClickHouse (prod) or the local scored jsonl. The SQL and the local
path produce identical dataset shapes.

  python -m p1_agent_eval.curation.curate --source file \
      --path p1_agent_eval/data/turns.jsonl --out p1_agent_eval/data/curated
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def load_from_file(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines()]


def load_from_clickhouse() -> list[dict]:
    from shared import config
    import clickhouse_connect
    host = config.CLICKHOUSE_URL.split("://")[-1].split(":")[0]
    client = clickhouse_connect.get_client(host=host, port=8123, database=config.CLICKHOUSE_DB)
    res = client.query(
        "SELECT trace_id, turn_index, user_text, agent_text, tool_name, tool_ok, "
        "model, label, confidence, reward, tags FROM turns ORDER BY trace_id, turn_index")
    cols = res.column_names
    return [dict(zip(cols, row)) for row in res.result_rows]


def build_eval_set(turns: list[dict], per_bucket: int = 40,
                   min_conf: float = 0.9) -> list[dict]:
    """High-confidence, stratified-by-(task_type, model) benchmark."""
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for t in turns:
        tags = t.get("tags") or {}
        task = tags.get("task_type", "unknown")
        if float(t.get("confidence", 0)) >= min_conf or t.get("label_source") == "ground_truth":
            buckets[(task, t["model"])].append(t)
    out = []
    rng = random.Random(0)
    for key, items in buckets.items():
        rng.shuffle(items)
        for t in items[:per_bucket]:
            out.append({"input": f"{t['user_text']}", "expected_label": t["label"],
                        "task_type": key[0], "model": key[1]})
    return out


def build_golden_paths(turns: list[dict]) -> list[dict]:
    """Traces where EVERY turn is good -> ideal trajectories for warm start."""
    by_trace: dict[str, list[dict]] = defaultdict(list)
    for t in turns:
        by_trace[t["trace_id"]].append(t)
    golden = []
    for tid, trace in by_trace.items():
        trace.sort(key=lambda x: x["turn_index"])
        if trace and all(t["label"] == "good" for t in trace):
            golden.append({
                "trace_id": tid,
                "messages": [m for t in trace for m in (
                    {"role": "user", "content": t["user_text"]},
                    {"role": "assistant", "content": t["agent_text"]})],
            })
    return golden


def build_dpo_pairs(turns: list[dict], max_pairs: int = 50000) -> list[dict]:
    """Mine (prompt, chosen, rejected) by matching good vs bad responses to the
    same user intent (task_type). chosen=good agent_text, rejected=bad."""
    good: dict[str, list[str]] = defaultdict(list)
    bad: dict[str, list[str]] = defaultdict(list)
    for t in turns:
        task = (t.get("tags") or {}).get("task_type", "unknown")
        if t["label"] == "good":
            good[task].append(t["agent_text"])
        elif t["label"] == "bad":
            bad[task].append(t["agent_text"])
    def best_match(prompt: str, pool: list[str], rng) -> str:
        """Prefer a response sharing the most keywords with the prompt so the
        (chosen, rejected) pair is topically coherent, not just same-bucket."""
        ptoks = set(prompt.lower().split())
        sample = rng.sample(pool, k=min(len(pool), 16))
        return max(sample, key=lambda r: len(ptoks & set(r.lower().split())))

    pairs, rng = [], random.Random(1)
    for t in turns:
        if len(pairs) >= max_pairs:
            break
        task = (t.get("tags") or {}).get("task_type", "unknown")
        if good[task] and bad[task]:
            pairs.append({
                "prompt": t["user_text"],
                "chosen": best_match(t["user_text"], good[task], rng),
                "rejected": best_match(t["user_text"], bad[task], rng),
            })
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="file", choices=["file", "clickhouse"])
    ap.add_argument("--path", default="p1_agent_eval/data/turns.jsonl")
    ap.add_argument("--out", default="p1_agent_eval/data/curated")
    ap.add_argument("--max-pairs", type=int, default=50000)
    args = ap.parse_args()

    turns = load_from_file(args.path) if args.source == "file" else load_from_clickhouse()
    print(f"loaded {len(turns)} scored turns")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    datasets = {
        "eval.jsonl": build_eval_set(turns),
        "golden_paths.jsonl": build_golden_paths(turns),
        "dpo_pairs.jsonl": build_dpo_pairs(turns, args.max_pairs),
    }
    for name, rows in datasets.items():
        p = out / name
        with p.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"  {name:20} {len(rows):>6} examples -> {p}")

    print("\ntriage-to-dataset complete. Point --source clickhouse at prod to run "
          "this continuously (cron/airflow).")


if __name__ == "__main__":
    main()
