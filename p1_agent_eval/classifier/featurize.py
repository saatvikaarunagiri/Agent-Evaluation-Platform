"""Turn -> model input text. Shared by both classifier backends so training
and inference featurize identically."""
from __future__ import annotations
from shared.schemas import Turn

LABELS = ["good", "needs_review", "bad"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


def turn_to_text(t: Turn | dict) -> str:
    """Flatten a turn into a single string the classifier reads.
    We include the tool name + ok flag because a failed tool call is a strong
    signal of a bad turn, and the model should see it."""
    d = t.__dict__ if isinstance(t, Turn) else t
    tool = d.get("tool_name", "")
    ok = "ok" if d.get("tool_ok", True) else "ERROR"
    return (f"[user] {d.get('user_text','')} "
            f"[agent] {d.get('agent_text','')} "
            f"[tool:{tool}={ok}]")
