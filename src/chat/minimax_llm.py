import string
from typing import Any

from langchain_openai import ChatOpenAI
from overrides import overrides

from src.chat.base import Llm
from src.prompt.base import BasePrompt


class MiniMaxLlm(Llm):


    def __init__(self, api_key, base_url, model_name="MiniMax-M3.0"):
        super().__init__()
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name


    @overrides
    def configLlm(self) -> ChatOpenAI:
        self.llm = ChatOpenAI(model_name=self.model_name,
                              openai_api_base=self.base_url,
                              openai_api_key=self.api_key)
        return self.llm


    @overrides
    def invokeLlm(self, input : BasePrompt, **kwargs:Any):
        self.llm.invoke(input=input, **kwargs)