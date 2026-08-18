
'''
提问类型：
    景点
    路线
    时间规划
    交通
    天气
    美食
'''
from enum import Enum


class QuestionType(Enum):
    ATTRACTION = (1, "景点"),
    ROUTE_LINE = (2, "路线"),
    SCHEDULE = (3, "日程"),
    TRAFFIC = (4, "交通"),
    WEATHER = (5, "天气"),
    FOOD = (6, "美食"),


class AnswerType(Enum):
    FULL = (1, "全量并发回答全部可能涉及的内容"),
    COT_ALL = (2, "根据用户问题,衍生全量按思考链回答"),
    COT_ONLY = (3, "只回答用户提问的相关内容,按用户提问顺序拆分问题后按顺序回答"),

