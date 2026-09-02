import uuid
from langchain_core.callbacks import BaseCallbackHandler
from overrides import overrides
from src.monitor.trace_context import current_trace_id


class TraceChainCallBack(BaseCallbackHandler):
    """
    在请求开始时设置 trace_id
    """
    def __init__(self, trace_id : int = None):
        self.trace_id = str(uuid.uuid4()) if trace_id is None else trace_id



    @overrides
    def on_chain_start(self, serialized, inputs, **kwargs):
        trace_id = self.trace_id
        current_trace_id.set(trace_id)
        # TODO: 这里还可以写 react_chains 表的初始记录
