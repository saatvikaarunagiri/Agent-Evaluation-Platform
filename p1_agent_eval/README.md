# Closed Loop Agent Evaluation & RL Platform

A pipeline that watches an LLM agent in production, **scores every turn** with a
cheap classifier (escalating only the genuinely ambiguous turns to an expensive
LLM judge), **auto-curates** eval and fine-tuning datasets from that stream, and
**closes the loop** with two-stage post-training (DPO → GRPO) whose reward signal
is the very same classifier. One source of truth for "was this a good turn,"
reused for monitoring, guardrails, and RL reward.

```
 agent ──OTel spans -> collector -> Kafka/Redpanda ─> ingest worker ─> ClickHouse
                                                             │ scores each turn
                                                             ▼
                                   ModernBERT classifier - low confidence -> LLM judge
                                        (cheap, <90ms)                      (slow, accurate)
                                                             │ label + reward
                                                             ▼
                                   curation ─> eval set · golden paths · DPO pairs
                                                             │
                                                             ▼
                                      DPO warm start.   ─>  online GRPO
                                      (prefer good traj.)    (reward = classifier)
```

---

## Why it's built this way

**The problem.** Evaluating an agent with an LLM judge on *every* turn is slow
and expensive. But most turns are easy to grade — only a minority are genuinely
ambiguous. So we train a small, fast classifier to handle the easy majority and
**route only low-confidence turns to the LLM**. That single idea drives the cost
down and gives us a reusable reward signal for RL.

**The pieces and the reasoning:**

| Choice | Why |
|---|---|
| **ClickHouse** over Postgres | Telemetry tags are high-cardinality (model, tool, tenant, prompt_version, task_type…). A columnar, vectorized engine aggregates across them far faster than a row store. Schema: [`infra/clickhouse/init.sql`](../infra/clickhouse/init.sql). |
| **Redpanda** over Kafka | Identical Kafka API, ~350 MB vs ~2 GB RAM. Your `confluent-kafka` code is unchanged. |
| **ModernBERT** classifier | A modern encoder (8k context, fast) fine-tuned for 3-way turn quality. Exports to **ONNX** for <90 ms inference on CPU / Apple Neural Engine (CoreML) — no Triton, no CUDA needed. |
| **Classifier as RL reward** | The same model gates quality *and* produces the GRPO reward, so "good turn" means one consistent thing across eval, guardrails, and training. |

---

## The ML, in detail

### 1. The data (`data/generate.py`)
Since you don't have your employer's production telemetry, we synthesize it.
Each **trace** is one user task = 4–24 **turns**. A turn carries the user text,
the agent's reply, which tool it called and whether that tool succeeded, plus
cost/latency and a hidden ground-truth quality label (`good` / `bad` /
`needs_review`).

Two deliberate design decisions make this a *real* ML problem, not a toy:
- **Good and bad turns share vocabulary.** Quality comes from subtle cues
  (hedging — "I think, maybe"; ignoring a tool error; looping; re-asking known
  info), not disjoint words. So a classifier must learn signal, not keywords.
- **~11% label noise is injected.** This caps the achievable F1 realistically
  (~0.85–0.91) and — crucially — *creates the ambiguous turns that should
  escalate to the LLM judge*. Without noise the routing logic would never fire.

```bash
python -m p1_agent_eval.data.generate --n 8000
# → writes p1_agent_eval/data/turns.jsonl, prints the label distribution
```

### 2. The classifier (`classifier/`)
Two interchangeable backends behind one `fit/predict` interface
(`classifier/model.py`):

- **`modernbert`** — `answerdotai/ModernBERT-base` fine-tuned as a 3-class
  sequence classifier. Training: AdamW, lr 2e-5, 2 epochs, max_len 256, batch 16.
  Runs on Apple **MPS** automatically. `--onnx` exports the graph (opset 17) for
  ONNX Runtime inference; on Apple Silicon ONNX RT uses the **CoreML** execution
  provider → the Neural Engine → the <90 ms target.
- **`tfidf`** — TF-IDF (1–2 grams) + balanced LogisticRegression. ~2 MB, trains
  in seconds, no PyTorch. Used by the offline demo, the tests, and as a perfectly
  good fallback if you don't want the ~1 GB torch download.

The **featurizer** (`classifier/featurize.py`) flattens a turn to
`[user] … [agent] … [tool:NAME=ok|ERROR]`. Including the tool-success flag matters:
a failed tool call is a strong predictor of a bad turn, and the model should see it.

```bash
python -m p1_agent_eval.classifier.train --backend tfidf          # fast
python -m p1_agent_eval.classifier.train --backend modernbert --onnx   # production path
```
**Reading the output:** you get a per-class precision/recall table plus
macro-F1 and weighted-F1. The `needs_review` class scores lowest (it's the
ambiguous set) — that's by design; those are the turns that escalate. Metrics
are saved to `classifier/artifacts/metrics.json` for the dashboard.

### 3. Low-confidence routing (`classifier/infer.py`, `classifier/judge.py`)
On every turn: `classifier.predict(turn)` returns a label and a **confidence**
(max softmax probability). If `confidence ≥ LOW_CONF_THRESHOLD` (default 0.75,
set in `.env`) we accept it. Otherwise we escalate to the **LLM judge**, which
tries, in order: Claude → OpenAI → local Ollama → a deterministic heuristic (so
it *always* works, even with zero keys). The scorer tracks the **escalation
rate**, which is exactly the cost lever:

```bash
python -m p1_agent_eval.classifier.infer 800
#   accuracy vs stored labels : 0.86
#   escalated to LLM judge     : 158 (19.8%)
#   => ~80% of turns never hit the LLM   (raise the threshold → fewer escalations)
```

### 4. Dataset curation (`curation/curate.py`)
The payoff of streaming telemetry: instead of a human spending days triaging
logs, we derive three datasets automatically and continuously:

- **`eval.jsonl`** — a balanced benchmark: only high-confidence turns (trustworthy
  labels), stratified by `(task_type, model)`.
- **`golden_paths.jsonl`** — full traces where *every* turn is `good`: ideal
  trajectories for the DPO/SFT warm start.
- **`dpo_pairs.jsonl`** — `(prompt, chosen, rejected)` preference pairs mined by
  matching a `good` vs a `bad` agent response to the same user intent. A
  keyword-overlap match keeps each pair topically coherent. This is the "50K+
  preference pairs" at scale.

```bash
python -m p1_agent_eval.curation.curate --source file     # or --source clickhouse in prod
```

### 5. Post-training (`posttrain/`)
Two stages, each with a **real TRL implementation** *and* a dependency-free
**simulation** so you can watch the objective converge without a GPU.

- **DPO** (`dpo.py`) — Direct Preference Optimization initializes the policy
  toward the `chosen` responses. Real path: TRL `DPOTrainer` + a small base model
  (`SmolLM2-135M`) + LoRA (r=8) so it fits in 16 GB, β=0.1. `--dry-run` runs the
  exact DPO loss (−log σ(β·margin)) on the preference pairs as a 1-parameter
  logistic model and prints the loss curve.
- **GRPO** (`grpo.py`) — Group Relative Policy Optimization is the *online* stage.
  For each prompt it samples a **group** of completions, scores each with the
  **classifier as reward model**, and pushes the policy toward above-average
  samples (advantage = reward − group mean) with a **KL penalty to the reference
  policy** (β=0.04) to limit off-policy drift. `--dry-run` runs this as a real
  policy-gradient loop over response "styles," scoring with the actual classifier,
  and reports task-success lift and — honestly — **drift measured as KL-with-penalty
  vs the same loop with the penalty off**.

```bash
python -m p1_agent_eval.posttrain.dpo  --dry-run
python -m p1_agent_eval.posttrain.grpo --dry-run --steps 80
# real training (downloads a ~135M model, uses MPS):
python -m p1_agent_eval.posttrain.dpo  --base HuggingFaceTB/SmolLM2-135M-Instruct
python -m p1_agent_eval.posttrain.grpo --base HuggingFaceTB/SmolLM2-135M-Instruct
```

---

## Run it

### Offline, no Docker, ~30s
```bash
make p1-demo          # generate → train → score → curate → DPO → GRPO, with a summary table
```

### The real streaming pipeline
```bash
make up-agent         # redpanda + clickhouse + otel collector  (docker)
make p1-data          # synthetic telemetry (or instrument your agent — see ingest/otel_agent.py)
make p1-train-clf     # classifier (add --backend modernbert --onnx for production)
make p1-ingest        # scores turns → ClickHouse  (edit --source/--sink flags inside)
make p1-curate        # auto-build datasets
make p1-dpo           # stage 1
make p1-grpo          # stage 2
```
Query ClickHouse while ingest runs:
```sql
SELECT model, count(), avg(latency_ms), sum(cost_usd), avg(reward)
FROM telemetry.turns GROUP BY model;
-- or the pre-aggregated rollup:
SELECT * FROM telemetry.turns_rollup_1m ORDER BY minute DESC LIMIT 20;
```

---

## How to read every metric

| Metric | Where | What it means | What's "good" |
|---|---|---|---|
| weighted-F1 | classifier train | overall classifier quality, class-weighted | tfidf ~0.85; ModernBERT target ~0.91 |
| needs_review F1 | classifier train | quality on the ambiguous class | *lower on purpose* — these escalate |
| escalation rate | `infer.py` | share of turns sent to the LLM | lower = cheaper; 8–20% here |
| cost saving | `infer.py` | `1 − escalation_rate` | ~80–92% of turns avoid the LLM |
| task-success lift | GRPO | prob. mass on good response styles, end vs start | positive and rising |
| drift reduction | GRPO | KL-to-reference with penalty vs without | the KL penalty keeps drift lower |

---

## From synthetic to real

1. **Telemetry** → instrument your agent with OpenTelemetry like
   `ingest/otel_agent.py` (set `agent.*` span attributes), point the collector at
   it, and run `make p1-ingest --source file` (span file) or `--source kafka`.
2. **Labels** → your first few thousand labels can come from the LLM judge; feed
   human corrections back as `label_source=human` (they're trusted over the
   classifier in curation).
3. **Classifier** → train with `--backend modernbert --onnx` on your labeled
   turns; serve the ONNX model from `infer.py`.
4. **RL** → run the real `DPOTrainer`/`GRPOTrainer` paths with your base model.
   The reward function already calls your trained classifier — no change needed.

---

## File guide
| File | Role |
|---|---|
| `data/generate.py` | synthetic multi-turn agent traces (stand-in for prod telemetry) |
| `ingest/otel_agent.py` | example agent emitting OTel spans |
| `ingest/worker.py` | source (kafka/file/replay) → classifier → sink (clickhouse/stdout) |
| `ingest/clickhouse_sink.py` | batched ClickHouse writer |
| `classifier/featurize.py` | turn → model input string |
| `classifier/model.py` | ModernBERT + tfidf backends, ONNX export |
| `classifier/train.py` | train + evaluate + save artifacts |
| `classifier/infer.py` | online scoring with low-confidence routing |
| `classifier/judge.py` | LLM judge with 4-way fallback |
| `curation/curate.py` | eval / golden-path / DPO-pair builders |
| `posttrain/dpo.py` | DPO (TRL + simulation) |
| `posttrain/grpo.py` | online GRPO (TRL + simulation), classifier reward |
| `demo.py` | the whole loop in one command |
