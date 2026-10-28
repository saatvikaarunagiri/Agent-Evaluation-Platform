-- =============================================================================
-- ClickHouse schema for agent telemetry (Project 1)
--
-- Design notes:
--  * One row per agent TURN. A "trace" (one user task) has many turns.
--  * Tags are high-cardinality (model, tool, tenant, prompt_version, ...).
--    We store them as a Map(String,String) so new tags need no migration,
--    and ClickHouse still aggregates across them vectorized.
--  * MergeTree ordered by (ts, trace_id) gives fast time-range + trace lookups.
--  * A materialized view keeps a rolling per-minute quality rollup so the
--    dashboard reads pre-aggregated data instead of scanning raw turns.
-- =============================================================================

CREATE DATABASE IF NOT EXISTS telemetry;

CREATE TABLE IF NOT EXISTS telemetry.turns
(
    ts              DateTime64(3)              DEFAULT now64(3),
    trace_id        String,
    turn_id         String,
    turn_index      UInt16,

    -- the actual content of the turn
    user_text       String,
    agent_text      String,
    tool_name       LowCardinality(String),
    tool_args       String,                    -- JSON string
    tool_ok         UInt8,                     -- 1 = tool call succeeded

    -- model / cost / latency
    model           LowCardinality(String),
    prompt_tokens   UInt32,
    completion_tokens UInt32,
    latency_ms      UInt32,
    cost_usd        Float64,

    -- quality signals (filled by the classifier, see classifier/)
    label           LowCardinality(String),    -- good | bad | needs_review | ''
    label_source    LowCardinality(String),    -- modernbert | llm_judge | human | ''
    confidence      Float32,                   -- classifier confidence 0..1
    reward          Float32,                   -- reward signal reused for RL

    -- free-form high-cardinality tags
    tags            Map(String, String)
)
ENGINE = MergeTree
PARTITION BY toYYYYMMDD(ts)
ORDER BY (ts, trace_id, turn_index)
TTL toDateTime(ts) + INTERVAL 90 DAY;        -- auto-expire old telemetry

-- Per-minute rollup: live quality + cost without scanning raw turns.
CREATE TABLE IF NOT EXISTS telemetry.turns_rollup_1m
(
    minute       DateTime,
    model        LowCardinality(String),
    turns        UInt64,
    bad          UInt64,
    needs_review UInt64,
    avg_latency  Float64,
    sum_cost     Float64,
    avg_reward   Float64
)
ENGINE = SummingMergeTree
ORDER BY (minute, model);

CREATE MATERIALIZED VIEW IF NOT EXISTS telemetry.turns_rollup_1m_mv
TO telemetry.turns_rollup_1m AS
SELECT
    toStartOfMinute(ts)                        AS minute,
    model,
    count()                                    AS turns,
    countIf(label = 'bad')                     AS bad,
    countIf(label = 'needs_review')            AS needs_review,
    avg(latency_ms)                            AS avg_latency,
    sum(cost_usd)                              AS sum_cost,
    avg(reward)                                AS avg_reward
FROM telemetry.turns
GROUP BY minute, model;
