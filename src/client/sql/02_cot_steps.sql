-- ============================================
-- 02. CoT 步骤表
-- 一个 ReAct 链路下 0-N 条 CoT 步骤
-- 用于后续做热点问题缓存基础
-- ============================================

CREATE TABLE IF NOT EXISTS cot_steps (
    -- 关联到 react_chains.trace_id
    trace_id String,

    -- 步骤序号（同一 trace_id 内单调递增）
    step_index Int16,

    -- CoT 三元组：思考 -> 行动 -> 观察
    thought String,                    -- LLM 的思考内容（可作为缓存 key）
    action String DEFAULT '',          -- 决定做什么（如调哪个 tool）
    observation Nullable(String),      -- tool/环境返回的观察

    -- 元数据
    duration_ms Nullable(Int32),
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
ORDER BY (trace_id, step_index)
PARTITION BY toYYYYMM(created_at)
SETTINGS index_granularity = 8192;