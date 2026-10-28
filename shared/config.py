"""Central config. Everything is overridable via environment variables so the
same code runs locally, in Docker, and in production with zero edits.
"""
import os


def _b(key: str, default: bool) -> bool:
    return os.getenv(key, str(default)).lower() in ("1", "true", "yes")


# Infra endpoints -----------------------------------------------------------
KAFKA_BROKERS   = os.getenv("KAFKA_BROKERS", "localhost:9092")
CLICKHOUSE_URL  = os.getenv("CLICKHOUSE_URL", "http://localhost:8123")
CLICKHOUSE_DB   = os.getenv("CLICKHOUSE_DB", "telemetry")
REDIS_URL       = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Topics --------------------------------------------------------------------
TOPIC_TURNS_RAW   = os.getenv("TOPIC_TURNS_RAW", "agent.turns.raw")
TOPIC_TURNS_SCORED= os.getenv("TOPIC_TURNS_SCORED", "agent.turns.scored")
TOPIC_TXNS        = os.getenv("TOPIC_TXNS", "payments.transactions")
TOPIC_FEATURES    = os.getenv("TOPIC_FEATURES", "payments.features")

# Model / API ---------------------------------------------------------------
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL   = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")

# Local model backends ------------------------------------------------------
# On Apple Silicon we prefer local/free backends. OLLAMA_URL serves the
# LLM-judge fallback if you don't want to pay any API.
OLLAMA_URL        = os.getenv("OLLAMA_URL", "http://localhost:11434")
USE_LOCAL_LLM     = _b("USE_LOCAL_LLM", True)     # default: free/local

# Classifier confidence threshold: below this, route turn back to the LLM judge
LOW_CONF_THRESHOLD = float(os.getenv("LOW_CONF_THRESHOLD", "0.75"))
