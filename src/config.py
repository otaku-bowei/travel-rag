"""
配置模块 - 旅游RAG项目配置
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# ============== 项目路径 ==============
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "knowledge"
VECTORSTORE_DIR = PROJECT_ROOT / "data" / "vectorstore"

# ============== LLM 配置 ==============
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "minimax")

# MiniMax 配置
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY", "")
MINIMAX_BASE_URL = "https://api.minimax.chat/v1"
MINIMAX_MODEL = os.getenv("MINIMAX_MODEL", "MiniMax-M2.1")

# OpenAI 配置（备用）
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ============== Embedding 配置 ==============
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")

# ============== RAG 配置 ==============
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K = int(os.getenv("TOP_K", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))

# ============== Prompt 配置 ==============
SYSTEM_PROMPT = """你是一个专业的旅游顾问助手，熟悉全球热门旅游目的地的景点、美食、交通和行程规划。
请根据用户的提问，结合知识库中的旅游攻略，给出详细的行程建议。
回答时需要包含：推荐景点、美食推荐、交通建议、注意事项等。
请用友好的语气回答，如果知识库中没有相关信息，请如实告知用户。"""

# ============== 向量库配置 ==============
COLLECTION_NAME = "travel_guide"

# 存储类型：chroma / elasticsearch
VECTORSTORE_TYPE = os.getenv("VECTORSTORE_TYPE", "chroma")

# Elasticsearch 配置
ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")
ES_INDEX = os.getenv("ES_INDEX", "travel_guide")
ES_USER = os.getenv("ES_USER", "")
ES_PASSWORD = os.getenv("ES_PASSWORD", "")

# ============== LangChain Serve 配置 ==============
LANGCHAIN_HOST = os.getenv("LANGCHAIN_HOST", "http://localhost:8000")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true") == "true"
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "http://localhost:1984")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "travel-rag")