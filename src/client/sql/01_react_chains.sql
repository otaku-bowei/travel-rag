-- ============================================
-- 01. ReAct 链路表
-- 一个用户请求对应一条记录
-- ============================================

CREATE TABLE IF NOT EXISTS react_chains (
    -- 主键 / 链路 ID（UUID）
    trace_id String,

    -- 用户上下文
    user_id String DEFAULT '',
    session_id String DEFAULT '',

    -- 请求内容
    user_question String,
    final_answer String DEFAULT '',

    -- 工具使用汇总（0 表示没用到工具）
    tool_call_count Int8 DEFAULT 0,
    tool_names Array(String) DEFAULT [],

    -- CoT 步骤数（0 表示直接回答）
    cot_step_count Int8 DEFAULT 0,

    -- LLM 元数据
    llm_model String DEFAULT '',
    prompt_tokens Nullable(Int32),
    completion_tokens Nullable(Int32),
    total_tokens Nullable(Int32),

    -- 时间
    started_at DateTime DEFAULT now(),
    ended_at DateTime DEFAULT now(),
    duration_ms Nullable(Int32),

    -- 状态
    success UInt8 DEFAULT 1,
    error_msg Nullable(String)
) ENGINE = MergeTree()
ORDER BY (started_at, trace_id)
PARTITION BY toYYYYMM(started_at)
SETTINGS index_granularity = 8192;