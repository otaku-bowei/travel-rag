import string
from typing import Sequence, Any, Callable

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool
from langgraph.graph import MessagesState
from overrides import overrides

from src.agent.base import Agent, mix_system_message
from src.chat.base import Llm
from src.chat.callback.tool_callback import ClickhouseRecordToolCallback
from src.chat.callback.traces_callback import TraceChainCallBack
from src.chat.callback.react_callback import ClickhouseRecordReactCallback
from src.chat.callback.cot_callback import ClickhouseRecordCoTCallback
from src.client import get_clickhouse_client
from src.prompt.base import BasePrompt


class CotAgent(Agent):


    def __init__(self,
                 llm : Llm,
                 tools: Sequence[dict[str, Any] | type | Callable | BaseTool] = None,
                 ):
        super().__init__()
        self._llm = llm
        self._build_agent(tools)


    def _build_agent(self, tools):
        """
                llm: 已经 bind_tools 的 ChatOpenAI
                tools: 工具列表
                """
        # 可选：加记忆（支持多轮对话）
        # memory = MemorySaver()
        agent = create_agent(
            model=self._llm.llm,
            tools=tools,
            # checkpointer=memory,
        )
        self.agent = agent

    @overrides
    def invoke(self, query: string, base_prompt: BasePrompt, **kwargs: Any) -> AIMessage:
        msgs = []
        sys_msgs = []
        sys_msgs.extend(base_prompt.get_messages())
        if len(base_prompt.get_kwargs_messages()) != 0:
            sys_msgs.extend(base_prompt.get_formatted_prompt(**kwargs))
        msgs.append(mix_system_message(sys_msgs))
        msgs.append(HumanMessage(content=query))
        # for i, m in enumerate(msgs):
        #     print(f"[{i}] type={type(m).__name__} content={repr(str(m.content))[:80]}")
        config = {
            "callbacks": [
                # 必须最先：设置 current_trace_id
                TraceChainCallBack(),
                # ClickhouseRecordReactCallback(get_clickhouse_client()),
                ClickhouseRecordCoTCallback(get_clickhouse_client()),
                ClickhouseRecordToolCallback(get_clickhouse_client()),
            ]
        }
        response = self.agent.invoke({"messages": msgs}, config=config)
        return response

    @overrides
    async def ainvoke(self, query, base_prompt, **kwargs):
        msgs = []
        sys_msgs = []
        sys_msgs.extend(base_prompt.get_messages())
        if len(base_prompt.get_kwargs_messages()) != 0:
            sys_msgs.extend(base_prompt.get_formatted_prompt(**kwargs))
        msgs.append(mix_system_message(sys_msgs))
        msgs.append(HumanMessage(content=query))
        config = {
            "callbacks": [
                TraceChainCallBack(),
                # ClickhouseRecordReactCallback(get_clickhouse_client()),
                ClickhouseRecordCoTCallback(get_clickhouse_client()),
                ClickhouseRecordToolCallback(get_clickhouse_client()),
            ]
        }
        response = await self.agent.ainvoke({"messages": msgs}, config=config)
        return response