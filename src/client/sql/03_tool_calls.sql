-- ============================================
-- 03. 工具调用表
-- 一个 ReAct 链路下 0-N 条工具调用
-- 0 条由 react_chains.tool_call_count = 0 标识
-- ============================================

CREATE TABLE IF NOT EXISTS tool_calls (
    -- 关联到 react_chains.trace_id
    trace_id String,

    -- 单次 tool 调用的唯一 ID（用于去重）
    tool_call_id String,

    -- 工具信息
    tool_name String,
    tool_input String,                  -- 输入参数（JSON 字符串）
    tool_output Nullable(String),       -- 输出结果（JSON 字符串，可能很大）

    -- 时间
    duration_ms Nullable(Int32),
    created_at DateTime DEFAULT now(),

    -- 状态
    success UInt8 DEFAULT 1,
    error_msg Nullable(String)
) ENGINE = MergeTree()
ORDER BY (trace_id, tool_call_id)
PARTITION BY toYYYYMM(created_at)
SETTINGS index_granularity = 8192;