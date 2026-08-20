

'''
简单测试Agent
'''
from http.client import responses

from dotenv import load_dotenv
from openai import api_key

from src.chat.minimax_llm import MiniMaxLlm
from src.chat.qwen_llm import QwenLlm
from src.prompt.cot_prompt import CotPrompt
from src.prompt.travel_prompt import TravelPrompt
from src.prompt.target_type import QuestionType
from src.service.config import *
from src.tools.search_tool import search

'''
实现根据提问类型控制提示词的方向
'''
def testPromptTemplate():
    bp = TravelPrompt(qt=[QuestionType.WEATHER])
    print(bp.get_formatted_prompt(places=["大阪", "京都"], dates=["2026-08-17", "2026-08-18"]))


def testLlmRequest():
    """初始化"""
    api_key = read_environment_config("MINIMAX_API_KEY")
    model_name = read_environment_config("MINIMAX_MODEL")
    base_url = read_environment_config("MINIMAX_BASE_URL")
    llm = MiniMaxLlm(api_key, base_url, model_name)
    """模拟用户行为"""
    user_question = "你好，看下大阪天气"
    """基于TravelPrompt对问题进行翻译""" # TODO--改为ai实现的CoT
    bp = TravelPrompt(qt=[QuestionType.WEATHER])
    responses = llm.invokeLlm(input=user_question, base_prompt=bp, places=["大阪", "京都"], dates=["2026-08-17", "2026-08-18"])
    print(responses)


def testLlmToolRequest():
    """初始化"""
    api_key = read_environment_config("MINIMAX_API_KEY")
    model_name = read_environment_config("MINIMAX_MODEL")
    base_url = read_environment_config("MINIMAX_BASE_URL")
    """绑定工具"""
    tool = [search]
    llm = MiniMaxLlm(api_key, base_url, model_name, tools=tool)
    """模拟用户行为"""
    user_question = "你好，看下大阪天气"
    """基于TravelPrompt对问题进行翻译""" # TODO--改为ai实现的CoT
    bp = TravelPrompt(qt=[QuestionType.WEATHER])
    responses = llm.invokeLlm(input=user_question, base_prompt=bp, places=["大阪", "京都"], dates=["2026-08-17", "2026-08-18"])
    print(responses)


def toolUse():
    answer = search.invoke("大阪 京都 2026年8月17日 8月18日 天气预报")
    print(answer)



def testOllama():
    cp = CotPrompt()
    base_url = read_environment_config("OLLAMA_BASE_URL")
    api_key = read_environment_config("OLLAMA_API_KEY")
    llm = QwenLlm(api_key, base_url, )
    response= llm.invokeLlm(input="你好", base_prompt=cp)
    print(response)


def testUsingOllamaForCot():
    cp = CotPrompt()
    # base_url = read_environment_config("OLLAMA_BASE_URL")
    # api_key = read_environment_config("OLLAMA_API_KEY")
    # llm = QwenLlm(api_key, base_url, )
    """初始化"""
    api_key = read_environment_config("MINIMAX_API_KEY")
    model_name = read_environment_config("MINIMAX_MODEL")
    base_url = read_environment_config("MINIMAX_BASE_URL")
    llm = MiniMaxLlm(api_key, base_url, model_name, )
    response = llm.invokeLlm(input="如果我近期去东京旅游，怎么安排比较好", base_prompt=cp)
    print(response)


if __name__ == "__main__":
    load_dotenv()
    # testPromptTemplate()
    # testLlmRequest()
    # testLlmToolRequest()
    # toolUse()
    # testOllama()
    testUsingOllamaForCot()