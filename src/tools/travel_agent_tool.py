from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from src.agent.travel_agent import TravelAgent
from src.cache.cache_llm import init_travel_llm, get_travel_llm
from src.prompt.target_type import QuestionType
from src.prompt.travel_prompt import TravelPrompt
from src.tools.rag_tool import rag_search
from src.tools.travel_param_tool import get_travel_param


@tool
def travel_agent_tool(input:str,
                      config:dict,
                      ) -> AIMessage:
    """
    当用户问题或者提问意图涉及旅游{某个地点的美食、景点、天气、交通、国情等}的时候，使用该agent回答用户的问题。
    使用该agent前，先使用{get_travel_param}工具从用户问题解析必要参数
    """
    # config = get_travel_param.invoke({"query": input})
    travel_llm = get_travel_llm([rag_search, ])
    bp = TravelPrompt(qt=[QuestionType.WEATHER, QuestionType.ATTRACTION])
    travel_agent = TravelAgent(travel_llm, tools=[rag_search])
    responses = travel_agent.invoke(input=input, base_prompt=bp, places=config['places'], dates=config['dates'])
    return responses