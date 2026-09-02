from contextvars import ContextVar

# 贯穿整个请求链路的 trace_id
current_trace_id: ContextVar[str] = ContextVar('current_trace_id', default='')