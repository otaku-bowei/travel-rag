import string
from typing import Any

from langchain_core.callbacks import StdOutCallbackHandler
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from overrides import overrides

from src.chat.base import Llm
from src.prompt.base import BasePrompt


class MiniMaxLlm(Llm):

    def __init__(self, api_key, base_url, model_name="MiniMax-M3.0", tools=None, llm=None):
        super().__init__()
        if tools is None:
            tools = []
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.tools = tools
        if llm is None:
            self.configLlm(self.tools)

    @overrides
    def configLlm(self, tools=None) -> ChatOpenAI:
        self.llm = ChatOpenAI(model_name=self.model_name,
                              openai_api_base=self.base_url,
                              openai_api_key=self.api_key)
        if tools is not None:
            self.llm = self.llm.bind_tools(tools=tools)
        return self.llm

    @overrides
    def invokeLlm(self, input: string, base_prompt: BasePrompt, **kwargs: Any) -> AIMessage:
        msgs = []
        msgs.extend(base_prompt.get_messages())
        if len(base_prompt.get_kwargs_messages()) != 0:
            msgs.extend(base_prompt.get_messages())
        msgs.extend([HumanMessage(content=input)])
        response = self.llm.invoke(msgs)
        return response

    def to_dict(self) -> dict:
        # 不保存 tools（无法序列化），只保存配置
        return {"api_key": self.api_key, "base_url": self.base_url, "model_name": self.model_name}

    @classmethod
    def from_dict(cls, config: dict, tools=None):
        # 从配置重建 llm，tools 单独传入
        return cls(
            api_key=config["api_key"],
            base_url=config["base_url"],
            model_name=config["model_name"],
            tools=tools
        )