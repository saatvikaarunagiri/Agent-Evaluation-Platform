# =============================================================================
# Agent Eval + RL — one-command helpers. Run `make` or `make help`.
# All python targets run from the repo root so `import shared...` resolves.
# =============================================================================
.DEFAULT_GOAL := help
SHELL := /bin/bash
PY := .venv/bin/python

.PHONY: help venv env doctor up-agent up-dash down clean \
        p1-data p1-ingest p1-train-clf p1-curate p1-dpo p1-grpo p1-demo \
        dashboard test

help:  ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	 awk 'BEGIN{FS=":.*?## "}{printf "  \033[1m%-16s\033[0m %s\n", $$1, $$2}'

# ---- setup ----
venv:  ## Create the Python virtualenv
	python3 -m venv .venv && .venv/bin/pip install -U pip
env:  ## Create .env from the example if missing
	@test -f .env || (cp .env.example .env && echo "created .env")
doctor:  ## Check your machine has what it needs
	$(PY) scripts/doctor.py

# ---- infra (docker profiles) ----
up-agent:  ## Start infra (redpanda, clickhouse, otel collector)
	docker compose --profile agent-eval up -d
up-dash:  ## Start Grafana
	docker compose --profile dash up -d
down:  ## Stop all containers (keeps data)
	docker compose down
clean:  ## Stop all containers AND delete data volumes
	docker compose down -v

# ---- the pipeline ----
p1-data:       ## Generate synthetic agent turns
	$(PY) -m p1_agent_eval.data.generate --n 8000
p1-train-clf:  ## Train the per-turn classifier (+ --backend modernbert --onnx for prod)
	$(PY) -m p1_agent_eval.classifier.train --backend tfidf
p1-ingest:     ## Score telemetry -> ClickHouse (edit source/sink flags)
	$(PY) -m p1_agent_eval.ingest.worker
p1-curate:     ## Auto-curate eval + fine-tune datasets
	$(PY) -m p1_agent_eval.curation.curate --source file
p1-dpo:        ## Stage 1 post-training: DPO on golden paths
	$(PY) -m p1_agent_eval.posttrain.dpo --dry-run
p1-grpo:       ## Stage 2 post-training: online GRPO
	$(PY) -m p1_agent_eval.posttrain.grpo --dry-run --steps 80
p1-demo:       ## End-to-end offline demo (no docker needed)
	$(PY) -m p1_agent_eval.demo

# ---- extras ----
dashboard:  ## Serve the metrics page at :9000
	$(PY) -m dashboard.server
test:  ## Run the fast unit tests
	$(PY) -m pytest -q
