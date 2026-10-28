# SETUP — agent-eval, step by step (with a fix for every step)

Gets the pipeline running on a Mac (works on Linux too), free. Each step has an
**if it fails** note.

## 1. Prerequisites

```bash
# Homebrew (if you don't have it)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python@3.12          # need Python >= 3.11
brew install --cask docker        # only needed for the live streaming path
# optional, free local LLM judge:
brew install ollama && ollama serve & ollama pull llama3.2:3b
```
**If it fails:** `brew: command not found` → `eval "$(/opt/homebrew/bin/brew shellenv)"`. `python3` wrong version → `which -a python3`, put `/opt/homebrew/bin` first on PATH.

## 2. Run the offline demo (no Docker, no keys)

```bash
make venv && source .venv/bin/activate
make env                      # creates .env (optional for the demo)
pip install -r requirements-ci.txt
make p1-demo
```
**If it fails:**
- `make: command not found` → `xcode-select --install`.
- `ModuleNotFoundError: shared` → run from the repo root with the venv active (the `make` targets do this for you).

## 3. Train the real ModernBERT scorer (optional, uses your GPU)

```bash
pip install -r p1_agent_eval/requirements.txt       # torch etc. (~1GB, once)
python -m p1_agent_eval.classifier.train --backend modernbert --onnx
```
**If it fails:**
- torch download is huge → let it finish once; or stay on the free `tfidf` backend (`--backend tfidf`) — the whole loop still runs.
- MPS out-of-memory → `--epochs 1`, or train on CPU (the model is small).

## 4. The live streaming pipeline (optional, needs Docker)

```bash
make up-agent         # redpanda + clickhouse + otel collector
make p1-data          # synthetic telemetry (or point OTel at your agent)
make p1-train-clf
make p1-ingest        # scores turns -> ClickHouse
make p1-curate && make p1-dpo && make p1-grpo
```
Query ClickHouse:
```sql
SELECT model, count(), avg(reward) FROM telemetry.turns GROUP BY model;
```
**If it fails:**
- `clickhouse connection refused` → `docker compose --profile agent-eval ps`; wait ~15s for healthy, or `make up-agent` again.
- `No classifier artifact found` on ingest → run `make p1-train-clf` first.

## 5. Keeping it free

Default is **$0** (tiny classifier + local tools). The LLM judge only runs on
low-confidence turns and falls back to a local/heuristic path. To use a hosted
judge instead, put a key in `.env`:
```bash
ANTHROPIC_API_KEY=...    # and set USE_LOCAL_LLM=false
```

## 6. Publish / deploy

```bash
# push to GitHub
gh repo create agent-eval --public --source=. --remote=origin --push
# or: git remote add origin <url> && git push -u origin main
```
The retrain jobs are plain CLIs — wire `make p1-curate` / `p1-grpo` to a GitHub
Actions `cron` or any scheduler. Serve the classifier in prod by exporting ONNX
(`--onnx`) and running it under ONNX Runtime (CPU is fine).

## Troubleshooting cheatsheet
| Symptom | Fix |
|---|---|
| `make: command not found` | `xcode-select --install` |
| `ModuleNotFoundError: shared` | run from repo root with the venv active |
| torch too big | use `--backend tfidf` — full loop still runs |
| `clickhouse` refused | `make up-agent`, wait ~15s |
| tests fail on fresh clone | `pip install -r requirements-ci.txt -r requirements-dev.txt && bash scripts/ci_smoke.sh` |
