'''
提示词基类

'''
import string
from abc import ABC
from typing import Any

from annotated_types.test_cases import cases
from langchain_core.messages import SystemMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from sympy.strategies.core import switch

from src.prompt.target_type import QuestionType

'''
定义提示词模版
'''
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

    def get_formatted_prompt(self, **kwargs) -> SystemMessage:
        messages = self.get_kwargs_messages()
        lines = []
        for msg in messages:
            content = msg.content
            # 简单替换 {variable} 格式的占位符
            for k, v in kwargs.items():
                placeholder = f"{{{k}}}"
                if placeholder in content:
                    content = content.replace(placeholder, str(v))
            lines.append(content)
        return SystemMessage("\r\n".join(lines))