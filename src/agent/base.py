import string
from abc import ABC
from typing import Any

from langchain_core.runnables import Runnable

from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import create_agent

from src.prompt.base import BasePrompt


class Agent(ABC):

    def __init__(self):
        pass


    def invoke(self, input: string, base_prompt: BasePrompt, **kwargs: Any) -> AIMessage:
        pass
