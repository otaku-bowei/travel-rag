from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from src.agent.travel_agent import TravelAgent
from src.cache.cache_llm import init_travel_llm, get_travel_llm
from src.prompt.target_type import QuestionType
from src.prompt.travel_prompt import TravelPrompt
from src.tools.rag_tool import rag_search


@tool
def travel_agent_tool(input:str, config:dict) -> AIMessage:
    """
    当用户问题涉及旅游的时候，使用该agent回答用户的问题
    """
    travel_llm = get_travel_llm([rag_search])
    bp = TravelPrompt(qt=[QuestionType.WEATHER, QuestionType.ATTRACTION])
    travel_agent = TravelAgent(travel_llm, tools=[rag_search])
    responses = travel_agent.invoke(input=input, base_prompt=bp, places=config['places'], dates=config['dates'])
    return responses