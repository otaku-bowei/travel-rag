'''
提示词基类

'''
import string
from abc import ABC
from jinja2 import Environment, StrictUndefined
from typing import Any

from annotated_types.test_cases import cases
from jinja2 import UndefinedError, StrictUndefined
from langchain_core.messages import SystemMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from sympy.strategies.core import switch

from src.prompt.target_type import QuestionType

'''
定义提示词模版
'''
_env = Environment(undefined=StrictUndefined)

class BasePrompt(ABC):

    def __init__(self):
        pass

    def set_messages(self):
        pass

    def set_kwargs_messages(self):
        pass

    def get_messages(self) -> list[SystemMessage]:
        pass

    def get_kwargs_messages(self) -> list[SystemMessage]:
        pass

    def get_formatted_prompt(self, **kwargs) -> list[SystemMessage]:
        messages = self.get_kwargs_messages()
        lines = []
        for msg in messages:
            try:
                tpl = _env.from_string(msg.content)
                content = tpl.render(**kwargs)
            except UndefinedError as e:
                raise ValueError(f"模板 '{msg.content[:50]}' 缺少参数: {e}")
            lines.append(SystemMessage(content))
        return lines