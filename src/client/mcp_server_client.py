from mcp.server import Server
from mcp.server.stdio import stdio_server
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
import asyncio

from src.agent.cot_agent import CotAgent
from src.cache.cache_llm import get_cot_llm
from src.tools.rag_tool import rag_search
from src.tools.travel_agent_tool import travel_agent_tool
from src.tools.travel_param_tool import get_travel_param

app = Server("mcp-client")
cot_llm = get_cot_llm([rag_search, get_travel_param, travel_agent_tool])
master_agent = CotAgent(cot_llm, tools=[rag_search, get_travel_param, travel_agent_tool])


@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "ask_travel":
        # LangChain agent 接管业务逻辑
        result = master_agent.invoke({"messages": [HumanMessage(content=arguments["question"])]})
        return result["messages"][-1].content


async def main():
    async with stdio_server() as (r, w):
        await app.run(r, w, app.create_initialization_options())

asyncio.run(main())