import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
import json
_tools_cache = None


def init_mcp_tools() -> list:
    """预加载 MCP 工具（顶层调用一次，缓存）"""
    global _tools_cache
    if _tools_cache is not None:
        return _tools_cache

    client = MultiServerMCPClient({
        "travel-param": {
            "url": "http://localhost:9999/mcp",
            "transport": "streamable-http"
        }
    })
    _tools_cache = asyncio.run(client.get_tools())
    print(f"✅ MCP tools 加载: {[t.name for t in _tools_cache]}")
    return _tools_cache


def use_mcp_sync(query: str, tool_name : str) -> dict:
    """同步调用 MCP 工具（业务函数，纯同步）"""
    tools = init_mcp_tools()
    parse_tool = next(t for t in tools if t.name == tool_name)
    return parse_tool.invoke({"query": query})

async def use_mcp_async(query: str, tool_name: str) -> dict:
    """异步调用 MCP 工具（在 async 函数里用）"""
    tools = init_mcp_tools()
    parse_tool = next(t for t in tools if t.name == tool_name)
    result = await parse_tool.ainvoke({"query": query})
    # MCP 协议返回 list[TextContent]，解构成 dict
    if isinstance(result, list) and result:
        first = result[0]
        if isinstance(first, dict) and "text" in first:
            return json.loads(first["text"])
    return result