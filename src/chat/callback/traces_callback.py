import uuid
from langchain_core.callbacks import BaseCallbackHandler
from overrides import overrides


class TraceChainCallBack(BaseCallbackHandler):
    """
    给每次调用分配独立 trace_id。
    不再依赖 current_trace_id / ContextVar。
    """

    def __init__(self, trace_id: str = None):
        self.trace_id = str(uuid.uuid4()) if trace_id is None else trace_id

    @overrides
    def on_chain_start(self, serialized, inputs, **kwargs):
        # ⭐ 最原始版本：什么都不做，self.trace_id 已经够用了
        pass