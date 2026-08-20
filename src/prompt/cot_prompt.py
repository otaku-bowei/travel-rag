'''
提示词基类

'''
import string
from abc import ABC
from typing import Any

from annotated_types.test_cases import cases
from langchain_core.messages import SystemMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from overrides import overrides
from sympy.strategies.core import switch

from src.prompt.base import BasePrompt
from src.prompt.target_type import QuestionType

'''
定义提示词模版
'''


class CotPrompt(BasePrompt):

    def __init__(self, ):
        super().__init__()
        self.messages = []
        self.set_messages()
        self.kwargs_messages = []
        self.set_kwargs_messages()

    '''
    核心模板
    '''

    def base_template(self) -> SystemMessage:
        # 让LLM做简单的问题拆解
        return SystemMessage(content="推理用户的这个问题，拆解成几个小问题")

    def response_format_template(self) -> SystemMessage:
        # 规范响应格式，方便CoT后取数据
        return SystemMessage(content="必须按JSON格式输出,如:{\"intent\":\"travel\",\"reasoning\":\"用户想知道大阪8月10号至8月17号的旅游攻略\",\"sub_questions\":[\"大阪旅游景点推荐\",\"大阪8月10号只8月17号天气\",\"大阪美食推荐\",\"大阪夏季旅游\"]}")


    def intent_recognition_template(self) -> SystemMessage:
        # 做简单的意图分析
        return SystemMessage(content="分析用户的问题意图")

    def few_shot_template(self) -> SystemMessage:
        # few-shot应用
        return SystemMessage(content="示例1:\r\n"
                                     "用户问题：推荐一下大阪8月10日至8月17日可以玩什么\r\n"
                                     "输出：{\"intent\":\"travel\",\"reasoning\":\"用户想知道大阪8月10号至8月17号的旅游攻略\",\"sub_questions\":[\"大阪旅游景点推荐\",\"大阪8月10号只8月17号天气\",\"大阪美食推荐\",\"大阪夏季旅游\"]}\r\n"
                                     "示例2:\r\n"
                                     "用户问题：大阪去京都怎么走\r\n"
                                     "输出：{\"intent\":\"travel\",\"reasoning\":\"用户想知道大阪去京都的路线\",\"sub_questions\":[\"大阪到京都交通路线\",\"大阪到京都的交通耗时\"]}\r\n"
                                     "示例3:\r\n"
                                     "用户问题：近期有什么基金值得买的\r\n"
                                     "输出：{\"intent\":\"finance\",\"reasoning\":\"用户希望能推荐一些近期的理财基金\",\"sub_questions\":[\"近期的股票新闻\",\"近期的交易数据\",\"股票大V看好的板块\"]}\r\n"
                             )

    '''
        getter & setter
        '''

    @overrides
    def set_messages(self):
        result = [self.base_template(), self.intent_recognition_template(), self.response_format_template(), self.few_shot_template()]
        self.messages = result

    @overrides
    def set_kwargs_messages(self):
        self.kwargs_messages = []

    @overrides
    def get_messages(self) -> list[SystemMessage]:
        return self.messages

    @overrides
    def get_kwargs_messages(self) -> list[SystemMessage]:
        return self.kwargs_messages