#!/usr/bin/env bash
# Build the minimal artifacts so the full test suite runs (nothing skips).
# Used by CI and reproducible locally:  bash scripts/ci_smoke.sh
set -euo pipefail

echo "::group::tiny data + classifier"
python -m p1_agent_eval.data.generate --n 2000
python -m p1_agent_eval.classifier.train --backend tfidf
echo "::endgroup::"

echo "smoke artifacts ready"
