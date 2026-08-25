import string
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from overrides import overrides
from langchain_ollama import ChatOllama
from src.chat.base import Llm
from src.monitor.prompt_input_aspect import log_llm_last_input_prompt
from src.prompt.base import BasePrompt

"""
本地ollama安装的qwen的llm小模型
"""
class QwenLlm(Llm):

    def __init__(self, api_key:str, base_url : str, model_name = "qwen", tools=None, llm=None):
        super().__init__()
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.tools = tools
        if llm is None :
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

    @overrides
    def to_dict(self) -> dict:
        return {"api_key": self.api_key, "base_url": self.base_url, "model_name": self.model_name, "tools": self.tools}

    @classmethod
    def from_dict(cls, config: dict, tools : None):
        return cls(
            api_key=config["api_key"],
            base_url=config["base_url"],
            model_name=config["model_name"],
            tools=tools,
        )