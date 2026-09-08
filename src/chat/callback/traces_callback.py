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
        # ⭐ 修复 Bug2：trace_id 继承父级，不覆盖
        existing = current_trace_id.get()
        if existing:
            # 父链路已经设置过 trace_id，自己就别覆盖了
            # 这样 master 和 inner 共用同一个 trace_id
            return
        # 自己就是最外层，用自己的 UUID
        current_trace_id.set(self.trace_id)
        # TODO: 这里还可以写 react_chains 表的初始记录
