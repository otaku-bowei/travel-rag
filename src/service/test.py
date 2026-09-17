

'''
简单测试Agent
'''
from dotenv import load_dotenv
import time
from pathlib import Path
from src.vector import file_util as fu
from src.vector.milvus_search_type import CollectionType
from src.client.milvus_client import get_milvus_client
from src.vector.embedding import embedding_by_baai
from src.agent.cot_agent import CotAgent
from src.agent.travel_agent import TravelAgent
from src.cache.cache_llm import *
from src.chat.minimax_llm import MiniMaxLlm
from src.chat.qwen_llm import QwenLlm
from src.client.clickhouse_client import ClickHouseClient
from src.client.fast_api_client import app
from src.client.milvus_client import (
    get_milvus_client,
    reset_milvus_client,
)
from src.mcp.base import start_mcp_in_thread
from src.mcp.mcp_client import init_mcp_tools
from src.prompt.cot_prompt import CotPrompt
from src.prompt.travel_prompt import TravelPrompt
from src.prompt.target_type import QuestionType
from src.service.config import *
from src.service.snowflake import get_snowflake
from src.tools.date_tool import get_date_info, parse_relative_date
from src.tools.rag_tool import travel_rag_search
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
    init_travel_llm([travel_rag_search])
    travel_llm = get_travel_llm([travel_rag_search])
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
    init_travel_llm([travel_rag_search])
    travel_llm = get_travel_llm([travel_rag_search])
    """基于TravelPrompt对问题进行翻译"""  # TODO--改为ai实现的CoT
    bp = TravelPrompt(qt=[QuestionType.WEATHER, QuestionType.ATTRACTION])
    travel_agent = TravelAgent(travel_llm, tools=[travel_rag_search])
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


def test_milvus_client():
    """Milvus 读写测试（最简版：只 insert + search）

    前提：travel_docs / travel_query_cache 已在 Attu 手动建好（含 HNSW_PQ 索引 + Load）
    字段名沿用 Attu 现状：create_at（拼错）/ update_at（拼错）
    """
    import random
    from datetime import datetime

    DIM = 768
    print("\n=== Milvus 读写测试（最简版）===\n")

    # ==================== 1. 连接 ====================
    print("[1] 连接 Milvus ...")
    reset_milvus_client()
    mc = get_milvus_client()
    assert mc is not None
    print(f"   ✅ 已连接")

    # ==================== 1.5 Load（索引加载到内存后才能 search）====================
    print("\n[1.5] Load collections ...")
    mc._mc.load_collection("travel_docs")
    mc._mc.load_collection("travel_query_cache")
    print(f"   ✅ 已 Load")

    def rand_vec(seed: int) -> list[float]:
        rng = random.Random(seed)
        return [rng.random() for _ in range(DIM)]

    def now() -> str:
        # Milvus TIMESTAMPTZ 字段期望 ISO 8601 字符串（带 timezone）
        return datetime.now(timezone.utc).isoformat()

    # ==================== 2. travel_docs 读写 ====================
    print("\n=== travel_docs ===")

    # 2.1 insert 5 条
    print("\n[2] insert 5 条（id 100-104）...")
    docs = [
        {"id": get_snowflake(), "content": f"京都景点_{i}", "vector": rand_vec(i),
         "source": "官网", "category": "京都", "create_at": now(), "update_at": now()}
        for i in range(100, 105)
    ]
    res = mc.insert("travel_docs", docs)
    print(f"   ✅ insert: {res['insert_count']} 条")

    # 重新 Load（让新数据可见）
    mc._mc.load_collection("travel_docs")

    # 2.2 search top-3
    print("\n[3] search top-3...")
    hits = mc.search("travel_docs", [rand_vec(100)], top_k=3,
                     output_fields=["content", "category"])
    print(f"   top-3: {[(h['id'], h['content'], round(h['distance'], 4)) for h in hits[0]]}")
    print(f"   len: {len(hits[0])}")

    # 2.3 search + filter
    print("\n[4] search + filter (category=='京都')...")
    hits = mc.search("travel_docs", [rand_vec(100)], top_k=5,
                     filter_expr='category == "京都"',
                     output_fields=["content", "category"])
    print(f"   命中: {[(h['id'], h['category']) for h in hits[0]]}")
    print(f"   len: {len(hits[0])}")

    # ==================== 3. travel_query_cache 读写 ====================
    print("\n=== travel_query_cache ===")

    # 3.1 insert 3 条
    print("\n[5] insert 3 条（id 200-202）...")
    caches = [
        {"id": i, "query": f"问题_{i}", "query_vector": rand_vec(i),
         "response": f"回答_{i}", "response_vector": rand_vec(i + 1000),
         "create_at": now(), "update_at": now()}
        for i in range(200, 203)
    ]
    res = mc.insert("travel_query_cache", caches)
    print(f"   insert: {res['insert_count']} 条")

    # 重新 Load（让新数据可见）
    mc._mc.load_collection("travel_query_cache")

    # 3.2 search 相似 query
    print("\n[6] search 相似 query（top-2）...")
    hits = mc.search("travel_query_cache", [rand_vec(200)], top_k=2,
                     vector_field="query_vector",
                     output_fields=["query", "response"])
    print(f"   top-2: {[(h['id'], h['query'], round(h['distance'], 4)) for h in hits[0]]}")
    print(f"   len: {len(hits[0])}")

    # 3.3 search 相似 response
    print("\n[7] search 相似 response（top-2）...")
    hits = mc.search("travel_query_cache", [rand_vec(1200)], top_k=2,
                     vector_field="response_vector",
                     output_fields=["query", "response"])
    print(f"   top-2: {[(h['id'], h['response'], round(h['distance'], 4)) for h in hits[0]]}")
    print(f"   len: {len(hits[0])}")

    # ==================== 4. 关闭 ====================
    print("\n[8] 关闭连接 ...")
    mc.close()

    print("\n=== ✅ Milvus 读写测试通过 ===\n")


def safe_truncate(s: str, max_bytes: int) -> str:
    """按字节截断（Milvus VarChar max_length 是字节限制，不是字符）
    
    避免切碎 UTF-8 多字节字符
    """
    if not s:
        return s
    encoded = s.encode("utf-8")
    if len(encoded) <= max_bytes:
        return s
    truncated = encoded[:max_bytes]
    # 从尾部去除不完整的 UTF-8 字符（后缀10xxxxxx 表示 continuation byte）
    while truncated and (truncated[-1] & 0xC0) == 0x80:
        truncated = truncated[:-1]
    return truncated.decode("utf-8", errors="ignore")


def import_doc_to_milvus(limit: int = 0):
    """把 data/knowledge/ 下所有 md 导入到 Milvus（双表设计）

    主表 travel_docs：完整正文 + content 向量
    元表 travel_docs_meta：标题 + title 向量 + 外联 travel_doc_id → 主表 id

    Args:
        limit: 限制导入条数，0 表示不限制（默认全量导入）
    """
    from datetime import datetime, timezone
    from tqdm import tqdm
    sf = get_snowflake()  # ⭐ 雪花 id 单例（避免每次重新创建）
    from src.vector.milvus_service import insert as milvus_insert  # ⭐ 走 service 层
    # ⭐ pymilvus Timestamptz 需要 ISO 8601 字符串（不是 datetime 对象）
    now_ts = datetime.now(timezone.utc).isoformat()
    # 1. 解析所有段落（提取 title / content / source / url / likes）
    titles, contents, main_metas, meta_metas = [], [], [], []
    md_files = list(Path("../../data/knowledge").glob("*.md"))
    print(f"📂 找到 {len(md_files)} 个 md 文件")
    if limit > 0:
        print(f"🔢 测试模式：仅导入前 {limit} 条")
    for md_file in md_files:
        paras = fu.split_paragraphs(md_file.read_text(encoding="utf-8"))
        for idx, para in enumerate(tqdm(paras, desc=f"解析 {md_file.name}", unit="段")):
            info = fu.parse_md_paragraph(para)
            if not info["content"]:
                continue
            # 没标题就用正文前 50 字顶替
            title = info["title"] or info["content"][:50]  # 兑底逻辑，下一步会按字节再截断
            titles.append(title)
            contents.append(info["content"])
            main_id = sf.next_id()  # 主表 id（雪花）
            main_metas.append({
                "id": main_id,
                "content": safe_truncate(info["content"], 2048),  # ⭐ 按字节截断
                "source": safe_truncate(info["source"], 256),    # ⭐ 按字节截断
                "category": safe_truncate(md_file.stem, 64),     # ⭐ 按字节截断
                "create_at": now_ts,
                "update_at": now_ts,
            })
            meta_metas.append({
                "id": sf.next_id(),               # ⚠️ meta 表 PK（不要漏）
                "title": safe_truncate(title, 64),              # ⭐ 按字节截断（Milvus 实际64）
                "likes": safe_truncate(info["likes"], 64),      # ⭐ 按字节截断
                "line": safe_truncate(str(idx), 256),           # ⭐ 按字节截断
                "travel_doc_id": main_id,         # ⚠️ 外键 → 主表 id
                "create_at": now_ts,
                "update_at": now_ts,
            })
            # ⭐ 达到 limit 立即退出（测试模式）
            if limit > 0 and len(contents) >= limit:
                break
        if limit > 0 and len(contents) >= limit:
            break

    if not contents:
        print("⚠️  没有有效段落")
        return 0
    print(f"\n✅ 解析完成: {len(contents)} 条有效段落")
    # 2. ⭐ 分批 embedding（避免 OOM）+ 进度条
    BATCH = 16
    def encode_with_progress(texts, label):
        vecs = []
        for i in tqdm(range(0, len(texts), BATCH), desc=f"embedding {label}", unit="batch"):
            batch = texts[i:i+BATCH]
            vecs.extend(embedding_by_baai(batch).tolist())
        return vecs
    title_vecs = encode_with_progress(titles, "title")
    content_vecs = encode_with_progress(contents, "content")
    # 3. 组装双表 docs
    main_docs = [{**m, "vector": v} for m, v in zip(main_metas, content_vecs)]
    meta_docs = [{**m, "id": sf.next_id(), "title_vector": v}
                 for m, v in zip(meta_metas, title_vecs)]
    # 4. 双表 insert（走 service 层）
    main_name = CollectionType.TRAVEL_DOCS.value[0]
    meta_name = CollectionType.TRAVEL_DOCS_META.value[0]
    print(f"\n📥 写入主表 {main_name}...")
    milvus_insert(CollectionType.TRAVEL_DOCS, main_docs)
    print(f"📥 写入元表 {meta_name}...")
    milvus_insert(CollectionType.TRAVEL_DOCS_META, meta_docs)
    print(f"\n✅ 双表导入完成: {len(main_docs)} 条")
    return len(main_docs)


if __name__ == "__main__":
    load_dotenv()
    mcp_thread = start_mcp_in_thread()
    init_mcp_tools()
    init_travel_llm([
        parse_relative_date, get_date_info,
        weather_search,
        travel_rag_search,
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
    # test_milvus_client()
    # import_doc_to_milvus()