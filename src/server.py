#!/usr/bin/env python3
"""
LangChain Server - Web UI 启动器
用法: langchain serve 或 python src/server.py
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 加载环境变量
load_dotenv()
os.environ.setdefault("HF_ENDPOINT", os.getenv("HF_ENDPOINT", "https://hf-mirror.com"))

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

# ============== 配置 ==============
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "knowledge"
VECTORSTORE_DIR = PROJECT_ROOT / "data" / "vectorstore"
COLLECTION_NAME = "travel_guide"

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "minimax")
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY", "")
MINIMAX_BASE_URL = "https://api.minimax.chat/v1"
MINIMAX_MODEL = os.getenv("MINIMAX_MODEL", "MiniMax-M2.1")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K = int(os.getenv("TOP_K", "5"))

# 系统提示词
SYSTEM_PROMPT = """你是一个专业的旅游顾问助手，熟悉全球热门旅游目的地的景点、美食、交通和行程规划。
请根据用户的提问，结合知识库中的旅游攻略，给出详细的行程建议。
回答时需要包含：推荐景点、美食推荐、交通建议、注意事项等。
请用友好的语气回答，如果知识库中没有相关信息，请如实告知用户。"""


def load_documents() -> list[Document]:
    """加载知识库文档"""
    data_path = Path(DATA_DIR)
    documents = []

    for file_path in data_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in {".txt", ".md"}:
            try:
                loader = TextLoader(str(file_path), encoding="utf-8")
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = str(file_path.name)
                documents.extend(docs)
            except Exception as e:
                print(f"加载失败: {file_path.name} - {e}")

    return documents


def get_vectorstore() -> Chroma:
    """获取或构建向量库"""
    print("🔄 初始化 Embedding 模型...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )

    # 检查向量库是否存在
    if (VECTORSTORE_DIR / "chroma.sqlite3").exists():
        print("📂 加载已有向量库...")
        return Chroma(
            client=chromadb.PersistentClient(path=str(VECTORSTORE_DIR)),
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
        )

    # 构建新向量库
    print("📂 ���载文档...")
    docs = load_documents()
    print(f"✅ 加载 {len(docs)} 个文档")

    print("✂️ 分割文档...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    split_docs = text_splitter.split_documents(docs)
    print(f"✅ 分割为 {len(split_docs)} 个文本块")

    print("🔨 构建向量库...")
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(VECTORSTORE_DIR),
    )
    print("✅ 向量库构建完成")

    return vectorstore


def create_chain() -> Runnable:
    """创建 LangChain LCEL chain"""
    # 初始化 LLM
    if LLM_PROVIDER == "minimax":
        llm = ChatOpenAI(
            model=MINIMAX_MODEL,
            base_url=MINIMAX_BASE_URL,
            api_key=MINIMAX_API_KEY,
            temperature=0.7,
            max_tokens=2000,
        )
    else:
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
        OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        llm = ChatOpenAI(
            model=OPENAI_MODEL,
            api_key=OPENAI_API_KEY,
            temperature=0.7,
            max_tokens=2000,
        )

    # 获取向量库
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    # 构建提示词
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("system", "旅游攻略信息：\n{context}"),
        ("user", "{question}"),
    ])

    # 创建 LCEL chain（简化版）
    def get_context(question):
        docs = retriever.invoke(question)
        if not docs:
            return "未找到相关攻略信息"
        return "\n\n".join([d.page_content if hasattr(d, 'page_content') else str(d) for d in docs])

    chain = (
        {
            "context": lambda x: get_context(x["question"]),
            "question": lambda x: x["question"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


# 全局 chain 实例
_chain = None


def get_chain() -> Runnable:
    """获取 chain（延迟初始化）"""
    global _chain
    if _chain is None:
        _chain = create_chain()
    return _chain


if __name__ == "__main__":
    # 测试
    print("🔄 初始化 LangChain Server...")
    chain = get_chain()

    # 测试问答
    question = "京都三日游怎么安排？"
    print(f"\n❓ 问题: {question}")
    print("\n📝 回答:")

    result = chain.invoke({"question": question})
    print(result)