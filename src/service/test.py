

'''
简单测试Agent
'''
from http.client import responses

from dotenv import load_dotenv
from openai import api_key

from src.cache.cache_llm import *
from src.chat.minimax_llm import MiniMaxLlm
from src.chat.qwen_llm import QwenLlm
from src.prompt.cot_prompt import CotPrompt
from src.prompt.travel_prompt import TravelPrompt
from src.prompt.target_type import QuestionType
from src.service.config import *
from src.tools.search_tool import search
from src.vector.chroma_service import ChromaService

'''
实现根据提问类型控制提示词的方向
'''
def test_prompt_template():
    bp = TravelPrompt(qt=[QuestionType.WEATHER])
    print(bp.get_formatted_prompt(places=["大阪", "京都"], dates=["2026-08-17", "2026-08-18"]))


def test_llm_request():
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


def test_llm_tool_request():
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


def tool_use():
    answer = search.invoke("大阪 京都 2026年8月17日 8月18日 天气预报")
    print(answer)



def test_ollama():
    cp = CotPrompt()
    base_url = read_environment_config("OLLAMA_BASE_URL")
    api_key = read_environment_config("OLLAMA_API_KEY")
    llm = QwenLlm(api_key, base_url, )
    response= llm.invokeLlm(input="你好", base_prompt=cp)
    print(response)


def test_using_ollama_for_cot():
    cp = CotPrompt()
    base_url = read_environment_config("OLLAMA_BASE_URL")
    api_key = read_environment_config("OLLAMA_API_KEY")
    model_name = read_environment_config("OLLAMA_MODEL_NAME")
    llm = QwenLlm(api_key, base_url, model_name)
    """初始化"""
    # api_key = read_environment_config("MINIMAX_API_KEY")
    # model_name = read_environment_config("MINIMAX_MODEL")
    # base_url = read_environment_config("MINIMAX_BASE_URL")
    # llm = MiniMaxLlm(api_key, base_url, model_name, )
    response = llm.invokeLlm(input="如果我近期去东京旅游，怎么安排比较好", base_prompt=cp)
    print(response)


def test_import_file_vector():
    cs = ChromaService("../../data/chroma")
    file_path = "../../data/knowledge/kyoto_osaka.md"
    cs.import_md_file(file_path)


def test_vector_search():
    _search = "大阪哪里好玩"
    cs = ChromaService("../../data/chroma")
    documents = cs.search(_search)
    for d in documents:
        print(d)



def test_rag_tool():
    # # 1. 先进行CoT分析
    # init_cot_llm()
    # cot_llm = get_cot_llm()
    user_question = "东京有什么好玩的"
    # user_question = "大阪有什么好玩的"
    init_travel_llm()
    travel_llm = get_travel_llm()
    """基于TravelPrompt对问题进行翻译"""  # TODO--改为ai实现的CoT
    bp = TravelPrompt(qt=[QuestionType.WEATHER])
    responses = travel_llm.invokeLlm(input=user_question, base_prompt=bp, places=["东京"],
                              dates=["2026-08-17", "2026-08-18"])
    print(responses)


if __name__ == "__main__":
    load_dotenv()
    # testPromptTemplate()
    # testLlmRequest()
    # testLlmToolRequest()
    # toolUse()
    # testOllama()
    # testUsingOllamaForCot()
    # test_import_file_vector()
    # test_vector_search()
    test_rag_tool()