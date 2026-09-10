

'''
简单测试Agent
'''
from http.client import responses

from dotenv import load_dotenv
from openai import api_key

from src.agent.cot_agent import CotAgent
from src.agent.travel_agent import TravelAgent
from src.cache.cache_llm import *
from src.chat.minimax_llm import MiniMaxLlm
from src.chat.qwen_llm import QwenLlm
from src.client.clickhouse_client import ClickHouseClient
from src.client.fast_api_client import app
from src.mcp.base import start_mcp_in_thread
from src.mcp.mcp_client import init_mcp_tools
from src.prompt.cot_prompt import CotPrompt
from src.prompt.travel_prompt import TravelPrompt
from src.prompt.target_type import QuestionType
from src.service.config import *
from src.tools.date_tool import get_date_info, parse_relative_date
from src.tools.rag_tool import rag_search
from src.tools.search_tool import search, weather_search
from src.tools.travel_agent_tool import travel_agent_tool
from src.tools.travel_param_tool import get_travel_param
from src.vector.chroma_service import ChromaService
import asyncio

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
    init_travel_llm([rag_search])
    travel_llm = get_travel_llm([rag_search])
    """基于TravelPrompt对问题进行翻译"""  # TODO--改为ai实现的CoT
    bp = TravelPrompt(qt=[QuestionType.WEATHER, QuestionType.ATTRACTION])
    responses = travel_llm.invokeLlm(input=user_question, base_prompt=bp, places=["东京"],
                              dates=["2026-08-17", "2026-08-18"])
    print(responses)


def test_clickhouse_client():
    """测试 ClickHouse 连接"""
    print("\n=== ClickHouse 连接测试 ===")
    client = ClickHouseClient()
    try:
        # 1. 查询版本
        version = client.query("SELECT version()")
        print(f"✅ 版本: {version[0][0]}")

        # 2. 查询数据库
        dbs = client.query("SHOW DATABASES")
        db_names = [d[0] for d in dbs]
        print(f"✅ 数据库: {db_names}")

        # 3. 查询 rag_traces 表是否存在及结构
        if "travel_rag" in db_names:
            tables = client.query("SHOW TABLES FROM travel_rag")
            print(f"✅ travel_rag 表: {[t[0] for t in tables]}")

            if any("rag_traces" in t for t in tables):
                cols = client.query("DESCRIBE TABLE travel_rag.rag_traces")
                print(f"✅ rag_traces 字段数: {len(cols)}")
                count = client.query("SELECT count() FROM travel_rag.rag_traces")
                print(f"✅ 数据条数: {count[0][0]}")
        else:
            print("⚠️ travel_rag 数据库不存在")
    except Exception as e:
        print(f"❌ 连接失败: {e}")
    finally:
        client.close()
        print("✅ 连接已关闭\n")

def test_agent():
    user_question = "大阪有什么好玩的"
    # user_question = "大阪有什么好玩的"
    init_travel_llm([rag_search])
    travel_llm = get_travel_llm([rag_search])
    """基于TravelPrompt对问题进行翻译"""  # TODO--改为ai实现的CoT
    bp = TravelPrompt(qt=[QuestionType.WEATHER, QuestionType.ATTRACTION])
    travel_agent = TravelAgent(travel_llm, tools=[rag_search])
    responses = travel_agent.invoke(input=user_question, base_prompt=bp, places=["大阪"],
                                     dates=["2026-08-17", "2026-08-18"])
    print(responses)


async def loop_talk_to_agent():
    print("启动agent小助手")
    cot_llm = get_cot_llm([
        # get_travel_param,
        # parse_relative_date, get_date_info,
        travel_agent_tool,
                           ])
    # bp = TravelPrompt(qt=[QuestionType.WEATHER, QuestionType.ATTRACTION])
    bp = CotPrompt()
    cot_agent = CotAgent(cot_llm, tools=[
        # parse_relative_date, get_date_info,
        travel_agent_tool,
                                         # get_travel_param,
                                         ])
    while True:
        try:
            user_input = input("> ").strip()
            print("等待agent响应。。。")
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        # 退出条件
        if user_input.lower() in ("q", "exit", "quit"):
            print("Bye!")
            break
        # 空输入跳过
        if not user_input:
            continue
        # 调用 agent
        try:
            responses = await cot_agent.ainvoke(query=user_input, base_prompt=bp)
            print(responses)
            # 取最后一条 AI 消息作为回答
            answer = responses["messages"][-1].content
            print(f"\n{answer}\n")
        except Exception as e:
            print(f"\n[错误] {e}\n")


def test_fastapi_client():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    load_dotenv()
    mcp_thread = start_mcp_in_thread()
    init_mcp_tools()
    init_travel_llm([
        parse_relative_date, get_date_info,
        weather_search,
        rag_search,
        search,])
    init_cot_llm([travel_agent_tool])
    # testPromptTemplate()
    # testLlmRequest()
    # testLlmToolRequest()
    # toolUse()
    # testOllama()
    # testUsingOllamaForCot()
    # test_using_ollama_for_cot()
    # test_import_file_vector()
    # test_vector_search()
    # test_rag_tool()
    # test_clickhouse_client()
    # test_agent()
    # asyncio.run(loop_talk_to_agent())
    test_fastapi_client()
