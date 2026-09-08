from langchain_tavily import TavilySearch

from src.service.config import read_environment_config
from langchain_core.tools import tool

@tool
def search(query: str) -> str:
    """使用Tavily搜索功能

        何时使用：
        - 用户提问需要使用搜索功能时使用

        Args:
            query: 用户的查询字符串，要包含搜索的内容

        Returns:
            相关搜索结果

        注意：
        - 比 LLM 训练数据更新更准确
        """
    tavily_api_key = read_environment_config("TAVILY_API_KEY")
    tool = TavilySearch(tavily_api_key=tavily_api_key)
    return tool.invoke(input=query)


# TODO--根据股票新闻做量化分析，以及数据分析



# TODO--数据库查询工具



# TODO--RAG检索流程改为使用工具



