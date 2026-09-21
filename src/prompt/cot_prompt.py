'''
提示词基类

'''
import string
from abc import ABC
from typing import Any

from annotated_types.test_cases import cases
from langchain_core.messages import SystemMessage, SystemMessage
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate, FewShotChatMessagePromptTemplate, \
    ChatPromptTemplate
from overrides import overrides
from sympy.strategies.core import switch

from src.prompt.base import BasePrompt
from src.prompt.target_type import QuestionType

'''
定义提示词模版
'''


class CotPrompt(BasePrompt):

    def __init__(self, session_memory: list[str] = None, ):
        super().__init__()
        self.messages = []
        self._can_add_cust = True
        self._session_memory = session_memory
        # self.set_messages()
        self.format = self.response_format_template()
        self.kwargs_messages = []
        self.set_kwargs_messages()

    '''
    核心模板
    '''

    def base_template(self) -> SystemMessage:
        # 让LLM做简单的问题拆解
        # return SystemMessage(content="推理用户的这个问题，拆解成几个小问题，必要时在分析后使用相关工具或其他agent回答")
        return SystemMessage(
            # content="推理用户的这个问题，拆解成旅游的{景点、美食、交通、天气、日程}问题，必要时在分析后使用相关工具或其他agent回答")
            content="你是agent路由器，只能挑选 tool 调用，绝对不许自己回答。必须调用一个 tool。")
        # content="你是agent路由器，只负责挑选tool调用。"
        #         "绝对不许自己直接回答用户的问题——每一轮必须调用一个tool。"
        #         "可用tool：\n"
        #         "- travel_agent_tool: 用户问题涉及旅游（地点/美食/景点/交通/天气/日程/国情等）"
        #         "- common_agent_tool: 不属于旅游的问题（闲聊/问候/常识/其他话题/元问题）"
        #         "选 tool 的依据只看当前问题，不要被历史上下文干扰。")

    def response_format_template(self) -> SystemMessage:
        # 规范响应格式，方便CoT后取数据
        return SystemMessage(
            content="JSON格式为:{\"intent\":\"\",\"reasoning\":\"\",\"sub_questions\":[]}")

    def intent_recognition_template(self) -> SystemMessage:
        # 做简单的意图分析
        return SystemMessage(content="分析用户的问题意图")

    def few_shot_template(self) -> SystemMessage:
        # few-shot应用--TODO
        # examples = [
        #     {"input": "推荐一下大阪8月10日至8月17日可以玩什么", "output": "{\"intent\":\"travel\",\"reasoning\":\"用户想知道大阪8月10号至8月17号的旅游攻略\",\"sub_questions\":[\"大阪旅游景点推荐\",\"大阪8月10号只8月17号天气\",\"大阪美食推荐\",\"大阪夏季旅游\"]}\r\n"},
        #     {"input": "今天天气", "output": "调用 weather_tool"},
        # ]
        # few_shot = FewShotChatMessagePromptTemplate(
        #     example_prompt=ChatPromptTemplate.from_messages([
        #         ("human", "{input}"),
        #         ("ai", "{output}"),
        #     ]),
        #     examples=examples,
        # )
        return SystemMessage(content="示例1:\r\n"
                                     "用户问题：推荐一下大阪8月10日至8月17日可以玩什么\r\n"
                                     "输出：{\"intent\":\"travel\",\"reasoning\":\"用户想知道大阪8月10号至8月17号的旅游攻略\",\"sub_questions\":[\"大阪旅游景点推荐\",\"大阪8月10号只8月17号天气\",\"大阪美食推荐\",\"大阪夏季旅游\"]}\r\n"
                             # "示例2:\r\n"
                             # "用户问题：大阪去京都怎么走\r\n"
                             # "输出：{\"intent\":\"travel\",\"reasoning\":\"用户想知道大阪去京都的路线\",\"sub_questions\":[\"大阪到京都交通路线\",\"大阪到京都的交通耗时\"]}\r\n"
                             # "示例3:\r\n"
                             # "用户问题：近期有什么基金值得买的\r\n"
                             # "输出：{\"intent\":\"finance\",\"reasoning\":\"用户希望能推荐一些近期的理财基金\",\"sub_questions\":[\"近期的股票新闻\",\"近期的交易数据\",\"股票大V看好的板块\"]}\r\n"
                             )

    def customized_format(self, format_match: str):
        return SystemMessage(content="JSON格式为:" + format_match)

    def session_memory(self):
        if not self._session_memory:
            return SystemMessage(content="")
        lines = []
        for turn in self._session_memory:
            # 兼容两种入参：list[dict] 带 role，或 list[str] 只有 content
            if isinstance(turn, dict):
                role = turn.get("role", "unknown")
                content = turn.get("content", "")
            else:
                role = "user"
                content = str(turn)
            lines.append(f"- {role}: {content}")
        text = "\n".join(lines)
        return SystemMessage(
            content=(
                f"[上下文记忆] 仅作背景理解：\n{text}\n\n"
                f"当前用户问题才是主题，不要把历史内容混入当前回答。"
            )
        )

    '''
        getter & setter
        '''

    @overrides
    def set_messages(self):
        result = [self.base_template(),
                  # self.intent_recognition_template(),
                  self.session_memory(),
                  # self.response_format_template(),
                  # TODO-定义格式
                  # self.format,
                  # self.few_shot_template(),
                  ]
        self.messages = result

    @overrides
    def set_kwargs_messages(self):
        self.kwargs_messages = []

    @overrides
    def get_messages(self) -> list[SystemMessage]:
        self.set_messages()
        return self.messages

    @overrides
    def get_kwargs_messages(self) -> list[SystemMessage]:
        return self.kwargs_messages

    def set_customized_format(self, input: str):
        self.format = self.customized_format(input)
