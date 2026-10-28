# Agent Evaluation + Reinforcement Learning

Score an AI agent's every turn, cheaply, then use those scores to **make the agent better**. Runs on your laptop in one command. No GPU, no API keys, no cloud bill.

<p>
<img alt="python" src="https://img.shields.io/badge/python-3.11+-blue">
<img alt="tests" src="https://img.shields.io/badge/tests-passing-brightgreen">
<img alt="runs on" src="https://img.shields.io/badge/runs%20on-a%20MacBook-black">
<img alt="cost" src="https://img.shields.io/badge/cost-%240-success">
<img alt="license" src="https://img.shields.io/badge/license-MIT-green">
</p>

---

## Try it!

```bash
make venv && source .venv/bin/activate   
pip install -r requirements-ci.txt       
make p1-demo                            
```

One command generates fake agent logs, trains the scorer, grades every turn, auto-builds training sets, and fine-tunes — start to finish in ~30s, printing its scores.

---

## What it does?

An LLM judge grading **every** agent turn is slow and expensive. So:

- **A cheap classifier** scores each turn (good / bad / needs-review).
- **Only the unsure ones** get escalated to the slow LLM judge.
- Those scores **auto-build** eval + fine-tuning datasets.
- The agent is **fine-tuned** on them — DPO to warm up, then online GRPO where the reward *is* the classifier.

```
agent turns ─> classifier (only if unsure) ─> LLM judge
                    │
                    └─> datasets ─> DPO ─> GRPO ─> better agent
```

---

## Use it like a tool

| I want to… | Run this |
|---|---|
| See everything work | `make p1-demo` |
| See the numbers on one page | `make dashboard` → http://localhost:9000 |
| Train the real ModernBERT scorer | `make p1-train-clf` then see the guide |
| Run the live streaming pipeline | `make up-agent` then `make p1-ingest` |
| Run the tests | `make test` |
| See every command | `make help` |
| Something broke? | [SETUP.md](SETUP.md) — a fix for every step |

📖 **Want the deep dive** (the models, the math, every metric)? → **[p1_agent_eval/README.md](p1_agent_eval/README.md)**

---

## Good to know

- **Free by default** — a tiny classifier + local tools. Plug in an API key only if you want a hosted LLM judge.
- **🪆 Nothing is required** — no torch? no Docker? no internet? It falls back to a lighter path and still runs the whole loop.
- **🧪 Numbers are real but synthetic** — the demo *measures* its scores on generated logs. Point it at your real agent telemetry (the guide shows where) for your real numbers.

<sub>Built to run on an M-series MacBook · MIT licensed · `make help` for the full menu</sub>
