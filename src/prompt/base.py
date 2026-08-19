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

    def get_messages(self) -> list[SystemMessage]:
        pass

    def get_formatted_prompt(self, **kwargs) -> str:
        pass