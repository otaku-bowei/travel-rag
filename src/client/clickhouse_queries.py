"""
RAG 监控查询 QL
存放所有查询语句，与 Python 代码分离
"""

# ==================== DDL 建表 ====================

CREATE_RAG_TRACES_TABLE = """
-- RAG 监控宽表
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
SETTINGS index_granularity = 8192
"""

# ==================== 性能统计 ====================

RAG_PERFORMANCE_STATS = """
SELECT
    count() as total_requests,
    avg(total_duration_ms) as avg_duration,
    avg(retriever_duration_ms) as avg_retriever_duration,
    avg(llm_duration_ms) as avg_llm_duration,
    avg(retrieved_docs_count) as avg_docs_retrieved,
    avg(total_tokens) as avg_tokens
FROM rag_traces
WHERE created_at >= now() - INTERVAL {days} DAY
"""

AVG_DURATION = """
SELECT avg(total_duration_ms)
FROM rag_traces
WHERE created_at >= now() - INTERVAL {days} DAY
  AND total_duration_ms IS NOT NULL
"""

# ==================== Tool 使用统计 ====================

TOOL_USAGE_STATS = """
SELECT
    arrayJoin(tool_calls) as tool,
    count() as call_count
FROM rag_traces
WHERE created_at >= now() - INTERVAL {days} DAY
GROUP BY tool
ORDER BY call_count DESC
"""

# ==================== 检索统计 ====================

RETRIEVER_STATS = """
SELECT
    retriever_used,
    count() as usage_count,
    avg(retrieved_docs_count) as avg_docs,
    avg(retriever_duration_ms) as avg_duration
FROM rag_traces
WHERE created_at >= now() - INTERVAL {days} DAY
  AND retriever_used != ''
GROUP BY retriever_used
ORDER BY usage_count DESC
"""

# ==================== 问题分析 ====================

QUESTION_KEYWORDS = """
SELECT
    arrayJoin(tokenizeVector(user_question)) as word,
    count() as cnt
FROM rag_traces
WHERE created_at >= now() - INTERVAL {days} DAY
GROUP BY word
ORDER BY cnt DESC
LIMIT {top_n}
"""

USER_QUESTIONS = """
SELECT trace_id, user_id, user_question, created_at
FROM rag_traces
ORDER BY created_at DESC
LIMIT {limit}
"""

# ==================== 失败分析 ====================

FAILURE_ANALYSIS = """
SELECT
    error_msg,
    count() as cnt
FROM rag_traces
WHERE created_at >= now() - INTERVAL {days} DAY
  AND success = 0
GROUP BY error_msg
ORDER BY cnt DESC
"""

# ==================== 用户行为 ====================

USER_BEHAVIOR = """
SELECT
    user_id,
    count() as request_count,
    avg(total_duration_ms) as avg_duration,
    sum(total_tokens) as total_tokens
FROM rag_traces
WHERE created_at >= now() - INTERVAL {days} DAY
  AND user_id != ''
GROUP BY user_id
ORDER BY request_count DESC
LIMIT {top_n}
"""

# ==================== 会话分析 ====================

SESSION_DETAIL = """
SELECT
    trace_id,
    user_question,
    tool_call_count,
    total_duration_ms,
    created_at
FROM rag_traces
WHERE session_id = '{session_id}'
ORDER BY created_at
"""

SESSION_STATS = """
SELECT
    session_id,
    count() as request_count,
    avg(total_duration_ms) as avg_duration
FROM rag_traces
WHERE session_id != ''
GROUP BY session_id
ORDER BY request_count DESC
LIMIT 20
"""

# ==================== Token 统计 ====================

TOKEN_STATS = """
SELECT
    llm_model,
    sum(prompt_tokens) as total_prompt_tokens,
    sum(completion_tokens) as total_completion_tokens,
    sum(total_tokens) as total_tokens,
    avg(total_tokens) as avg_tokens,
    count() as request_count
FROM rag_traces
WHERE created_at >= now() - INTERVAL {days} DAY
  AND total_tokens IS NOT NULL
GROUP BY llm_model
ORDER BY request_count DESC
"""

# ========================记录tool使用
CLICKHOUSE_TOOL_RECORD = """
INSERT INTO tool_calls 
    (trace_id, tool_call_id, tool_name, tool_input, tool_output, duration_ms, success, error_msg) 
VALUES
 ('trace-uuid', 'call-uuid', 'rag_search', '{"query":"京都"}', '["doc1","doc2"]', 250, 1, '');
"""



