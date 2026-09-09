from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from src.agent.travel_agent import TravelAgent
from src.cache.cache_llm import init_travel_llm, get_travel_llm
from src.mcp.mcp_client import use_mcp_sync, use_mcp_async
from src.prompt.target_type import QuestionType
from src.prompt.travel_prompt import TravelPrompt
from src.tools.date_tool import get_date_info, parse_relative_date
from src.tools.rag_tool import rag_search
from src.tools.search_tool import search, weather_search
from src.tools.travel_param_tool import get_travel_param


@tool
async def travel_agent_tool(query:str,
                      # places: list[str] = None,
                      # dates: list[str] = None,
                      ) -> AIMessage:
    """这是一个可执行的agent。

            何时使用：
            - 当用户问题或者提问意图涉及旅游{某个地点的美食、景点、天气、交通、国情等}的时候，使用该agent回答用户的问题。

            Args:
                query: 用户的子问题
                # places: get_travel_param的响应结果，地点列表
                # dates: get_travel_param的响应结果，日期列表

            Returns:
                相关搜索结果

            注意：
            - 比 LLM 训练数据更新更准确
            - 使用该agent前，先使用{get_travel_param}工具从用户问题解析必要参数，然后使用该agent回答用户的问题
            """
    # config = await use_mcp_async(query, "parse_travel_params")
    config = await use_mcp_async(query, "parse_travel_params")
    places = config.get("places", [])
    dates = config.get("dates", [])
    travel_llm = get_travel_llm([
        parse_relative_date, get_date_info,
        weather_search,
        rag_search,
        search,
    ])
    bp = TravelPrompt(qt=[QuestionType.WEATHER, QuestionType.ATTRACTION, QuestionType.FOOD, QuestionType.SCHEDULE])
    travel_agent = TravelAgent(travel_llm, tools=[
        parse_relative_date, get_date_info,
        weather_search,
        rag_search,
        search,
        ])
    responses = await travel_agent.ainvoke(query=query, base_prompt=bp, places=places, dates=dates)
    if isinstance(responses, dict):
        return AIMessage(responses["messages"][-1].content)
    if hasattr(responses, "content"):
        return AIMessage(responses.content)
    return AIMessage(str(responses))