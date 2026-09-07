"""
MCP Server: travel-param
把 get_travel_param 工具暴露成 MCP 协议

启动方式：
  stdio:   python travel_param_server.py
  HTTP:    python travel_param_server.py --transport http --port 8000
"""
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

from src.tools.travel_param_tool import get_travel_param


# 初始化 MCP Server
mcp = FastMCP("travel-param", host="0.0.0.0", port=9999)


@mcp.tool()
async def parse_travel_params(query: str) -> dict:
    """从用户的旅游问题中解析地点、日期、交通方式、美食、路线等信息。

    何时使用：
    - 需要从自然语言问题中提取结构化旅游参数时
    - 在调用 travel_agent_tool 前，先用此工具提取 places 和 dates

    Args:
        query: 用户的旅游问题，如"推荐大阪8月10日至8月17日玩法"

    Returns:
        字典包含：
        - places: 地点列表，如 ["大阪", "京都"]
        - dates: 日期列表，如 ["2026-08-10", "2026-08-17"]
        - weather: 天气要求，可为空
        - traffic: 交通方式，如 "JR线"
        - foods: 美食列表，如 ["汤咖喱", "和牛"]
        - line: 路线，可为空
    """
    load_dotenv()
    response = get_travel_param.invoke({"query": query})
    print(f"[MCP] response: {response}")
    return response
