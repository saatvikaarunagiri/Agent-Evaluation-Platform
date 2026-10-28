# Contributing

Thanks for taking a look! See the [README](README.md) for what this is.

## Dev setup
```bash
make venv && source .venv/bin/activate
pip install -r requirements-ci.txt -r requirements-dev.txt
bash scripts/ci_smoke.sh      # builds tiny data + classifier so all tests run
make test
```

## Conventions
- **Python 3.11+**, linted with [ruff](https://docs.astral.sh/ruff/) (`ruff check .`).
- Keep the **light fallback working** — the pipeline must still run without torch,
  Docker, GPUs, or API keys. Guard heavy deps with a try/except and a documented
  fallback (see `p1_agent_eval/classifier/model.py` and `classifier/judge.py`).
- New behavior needs a fast test under `p1_agent_eval/tests/` (no network, no Docker).

## PRs
Branch off `main`, keep it focused, make sure `make test` passes (CI runs it on
3.11 and 3.12).
