import string
from abc import ABC
from typing import Any

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from src.prompt.base import BasePrompt


class Llm(ABC):
    def __init__(self):
        self.llm = None

    """配置大模型"""
    def configLlm(self) -> ChatOpenAI:
        pass

    """调用 LLM 并返回结果"""
    def invokeLlm(self, input: string, base_prompt: BasePrompt, **kwargs: Any) -> AIMessage:
        pass


    def to_dict(self) -> dict:
        pass

    @classmethod
    def from_dict(cls, config : dict, tools : None):
        pass
