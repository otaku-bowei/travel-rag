from langchain_tavily import TavilySearch

from src.service.config import read_environment_config
from langchain_core.tools import tool

@tool
def search(input: str) -> str:
    """使用Tavily搜索功能"""
    tavily_api_key = read_environment_config("TAVILY_API_KEY")
    tool = TavilySearch(tavily_api_key=tavily_api_key)
    return tool.invoke(input=input)


# TODO--根据股票新闻做量化分析，以及数据分析



# TODO--数据库查询工具



# TODO--RAG检索流程改为使用工具



