import string
from abc import ABC
from typing import Any

from langchain_openai import ChatOpenAI


class Llm(ABC):
    def __init__(self):
        self.llm = None


    def configLlm(self) -> ChatOpenAI:
        pass

    def invokeLlm(self, input : string, **kwargs:Any):
        pass