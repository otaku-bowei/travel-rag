-- RAG 监控宽表
-- 使用方式: clickhouse-client < rag_traces.sql

CREATE TABLE IF NOT EXISTS rag_traces (
    -- 基础信息
    trace_id String,
    user_id String,
    session_id String,

    -- 用户问题
    user_question String,
    question_length Int16,

    -- Tool 使用情况
    tool_calls Array(String),
    tool_call_count Int8,

    -- Tool 返回结果
    tool_names Array(String),
    tool_results Array(String),

    -- CoT 推理过程
    cot_thoughts Array(String),
    cot_steps Int8,

    -- RAG 检索情况
    retriever_used String,
    retrieved_docs_count Int8,
    retrieved_doc_ids Array(String),
    retrieved_doc_scores Array(Float32),
    top_k Int8,
    rerank_used UInt8,

    -- 最终结果
    final_answer String,
    answer_length Int32,

    -- LLM 使用情况
    llm_model String,
    prompt_tokens Nullable(Int32),
    completion_tokens Nullable(Int32),
    total_tokens Nullable(Int32),

    -- 性能指标
    total_duration_ms Nullable(Int32),
    retriever_duration_ms Nullable(Int32),
    llm_duration_ms Nullable(Int32),

    -- 状态
    success UInt8,
    error_msg Nullable(String),

    -- 元数据
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(created_at)
ORDER BY (created_at, trace_id)
SETTINGS index_granularity = 8192;
