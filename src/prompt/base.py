'''
提示词基类

'''
import string
from abc import ABC

from langchain_core.messages import SystemMessage, SystemMessage
from langchain_core.prompts import PromptTemplate

from src.prompt.target_type import QuestionType

'''
定义提示词模版
'''
class BasePrompt(ABC, PromptTemplate):

    def __init__(self, qt : list[QuestionType]):
        self.qt = qt

    '''
    getter & setter
    '''
    def getQuestionTypes(self):
        return self.qt


    '''
    target work
    核心模板
    '''
    def baseTemplate(self) -> SystemMessage:
        return SystemMessage(content="你是一个旅游AI助手。负责帮人从景点、路线、时间规划、交通、天气、美食这几个方面来规划旅游计划")

    def weatherTemplate(self, places : list[string], dates : list[string] = ["2026-08-17"]) -> SystemMessage:
        return SystemMessage(content="请查询下{places}在{dates}的天气情况。")

    def foodTemplate(self, places : list[string]) -> SystemMessage:
        return SystemMessage(content="请查询下{places}的美食。")

    def trafficTemplate(self, start, end : string) -> SystemMessage:
        return SystemMessage(content="请查询下{}到{}的交通方式以及交通路线，以及交通所需要的时间。")

    def attractionTemplate(self, places : list[string], dates : list[string] = None) -> SystemMessage:
        return SystemMessage(content="请列出{places}适合在{dates}游玩的景点，按推荐度从高到低排序，并给出推荐理由。")

    def routeLineTemplate(self, places : list[string]) -> SystemMessage:
        return SystemMessage(content="请列出{places}的游玩路线，并给出推荐游玩时长和理由。")

    def scheduleTemplate(self, places : list[string], days : int = None) -> SystemMessage:
        return SystemMessage(content="请列出{places}使用{days}天游玩日程，并给出理由。")

    