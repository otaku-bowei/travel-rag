from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from src.agent.common_agent import CommonAgent
from src.agent.travel_agent import TravelAgent
from src.cache.cache_llm import init_travel_llm, get_travel_llm, get_common_llm
from src.mcp.mcp_client import use_mcp_sync, use_mcp_async
from src.prompt.target_type import QuestionType
from src.prompt.travel_prompt import TravelPrompt
from src.tools.date_tool import get_date_info, parse_relative_date
from src.tools.rag_tool import travel_rag_search
from src.tools.search_tool import search, weather_search
from src.tools.travel_param_tool import get_travel_param


@tool
async def common_agent_tool(query:str,
                      ) -> AIMessage:
    """通用agent。

            何时使用：
            - 该agent使用优先级最低，其他agent都不匹配时使用该agent

            Args:
                query: 用户的问题

            Returns:
                相关搜索结果

            注意：
            - 比 LLM 训练数据更新更准确
            - 仅仅在其他tool都没执行时才使用该agent
            """
    common_llm = get_common_llm([
        search,
    ])
    common_agent = CommonAgent(common_llm, tools=[
        search,
        ])
    responses = await common_agent.ainvoke(query=query, base_prompt=None,)
    if isinstance(responses, dict):
        return AIMessage(responses["messages"][-1].content)
    if hasattr(responses, "content"):
        return AIMessage(responses.content)
    return AIMessage(str(responses))