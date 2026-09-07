import string
from abc import ABC
from typing import Any

from langchain_core.runnables import Runnable

from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.agents import create_agent

from src.prompt.base import BasePrompt


def mix_system_message(msgs: list[SystemMessage]) -> SystemMessage:
    if not msgs:
        return SystemMessage(content="")
    combined_content = "\n\n".join([m.content for m in msgs])
    return SystemMessage(content=combined_content)


class Agent(ABC):

    def __init__(self):
        pass


    def invoke(self, query: string, base_prompt: BasePrompt, **kwargs: Any) -> AIMessage:
        pass

    async def ainvoke(self, query, base_prompt, **kwargs):
        pass

