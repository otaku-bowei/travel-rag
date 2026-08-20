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
class TravelPrompt(BasePrompt):

    def __init__(self, qt: list[QuestionType]):
        super().__init__()
        self.messages = []
        self.qt = qt
        self.set_messages()
        self.kwargs_messages = []
        self.set_kwargs_messages()

    '''
    核心模板
    '''
    def base_template(self) -> SystemMessage:
        return SystemMessage(content="你是一个旅游AI助手。负责帮人从景点、路线、时间规划、交通、天气、美食这几个方面来规划旅游计划")

    def weather_template(self) -> SystemMessage:
        return SystemMessage(content="请查询下{places}在{dates}的天气情况。")

    def food_template(self) -> SystemMessage:
        return SystemMessage(content="请查询下{places}的美食。")

    def traffic_template(self) -> SystemMessage:
        return SystemMessage(content="请查询下{start}到{end}的交通方式以及交通路线，以及交通所需要的时间。")

    def attraction_template(self) -> SystemMessage:
        return SystemMessage(content="请列出{places}适合在{dates}游玩的景点，按推荐度从高到低排序，并给出推荐理由。")

    def route_line_template(self) -> SystemMessage:
        return SystemMessage(content="请列出{places}的游玩路线，并给出推荐游玩时长和理由。")

    def schedule_template(self) -> SystemMessage:
        return SystemMessage(content="请列出{places}使用{days}天游玩日程，并给出理由。")

    '''
        getter & setter
        '''

    def get_questionTypes(self):
        return self.qt


    @overrides
    def set_messages(self):
        result = [self.base_template()]
        for i in self.qt:
            match i:
                case QuestionType.WEATHER:
                    result.append(self.weather_template())
                case QuestionType.FOOD:
                    result.append(self.food_template())
                case QuestionType.ATTRACTION:
                    result.append(self.attraction_template())
                case QuestionType.TRAFFIC:
                    result.append(self.traffic_template())
                case QuestionType.SCHEDULE:
                    result.append(self.schedule_template())
                case QuestionType.ROUTE_LINE:
                    result.append(self.route_line_template())
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
