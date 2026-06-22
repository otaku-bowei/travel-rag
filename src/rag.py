"""
RAG核心模块 - 向量化 + 检索
支持 Chroma 和 Elasticsearch
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 设置 HuggingFace 镜像
os.environ.setdefault("HF_ENDPOINT", os.getenv("HF_ENDPOINT", "https://hf-mirror.com"))
from typing import List, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma, ElasticsearchStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from elasticsearch import Elasticsearch

import os
from .config import (
    EMBEDDING_MODEL,
    EMBEDDING_DEVICE,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    COLLECTION_NAME,
    VECTORSTORE_DIR,
    VECTORSTORE_TYPE,
    ES_HOST,
    ES_INDEX,
)
from .loader import TravelDocLoader, create_sample_data


# 支持的向量存储类型
VECTORSTORE_TYPES = {
    "chroma": Chroma,
    "elasticsearch": ElasticsearchStore,
}


class TravelRAG:
    """旅游RAG检索系统"""

    def __init__(
        self,
        data_dir: str,
        vectorstore_dir: Optional[str] = None,
    ):
        self.data_dir = Path(data_dir)
        self.vectorstore_dir = Path(vectorstore_dir) if vectorstore_dir else VECTORSTORE_DIR

        # 初始化Embedding模型
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )

        # 文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", "。", ".", " ", ""],
        )

        # 向量数据库
        self.vectorstore: Optional[object] = None

    def load_and_split(self) -> List[Document]:
        """加载并分割文档"""
        print("📂 加载文档...")
        loader = TravelDocLoader(str(self.data_dir))
        documents = loader.load_directory()

        print("✂️ 分割文档...")
        docs = self.text_splitter.split_documents(documents)
        print(f"✅ 分割完成，共 {len(docs)} 个文本块")

        return docs

    def _build_chroma(self, docs: List[Document], force_rebuild: bool = False):
        """构建Chroma向量库"""
        if not force_rebuild and self.vectorstore_dir.exists():
            if (self.vectorstore_dir / "chroma.sqlite3").exists():
                print("📂 加载已有Chroma向量库...")
                self.vectorstore = Chroma(
                    embedding_function=self.embeddings,
                    persist_directory=str(self.vectorstore_dir),
                )
                return

        print("🔨 构建Chroma向量数据库...")
        self.vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=str(self.vectorstore_dir),
        )
        print("✅ Chroma向量库构建完成")

    def _build_elasticsearch(self, docs: List[Document], force_rebuild: bool = False):
        """构建Elasticsearch向量库"""
        print("🔨 构建Elasticsearch向量数据库...")

        # 检查索引是否已存在
        try:
            es = Elasticsearch(ES_HOST)
            if es.indices.exists(index=ES_INDEX):
                if force_rebuild:
                    print(f"🗑️ 删除已有索引: {ES_INDEX}")
                    es.indices.delete(index=ES_INDEX)
                else:
                    print(f"📂 加载已有ES索引: {ES_INDEX}")
                    self.vectorstore = ElasticsearchStore(
                        embedding=self.embeddings,
                        index_name=ES_INDEX,
                        es_url=ES_HOST,
                    )
                    return
        except Exception as e:
            print(f"⚠️ ES连接检查: {e}")

        # 创建新索引
        self.vectorstore = ElasticsearchStore.from_documents(
            documents=docs,
            embedding=self.embeddings,
            index_name=ES_INDEX,
            es_url=ES_HOST,
        )
        print("✅ Elasticsearch向量库构建完成")

    def build_vectorstore(self, force_rebuild: bool = False):
        """构建向量数据库"""
        store_type = VECTORSTORE_TYPE.lower()

        if store_type == "chroma":
            docs = self.load_and_split()
            self._build_chroma(docs, force_rebuild)
        elif store_type == "elasticsearch":
            docs = self.load_and_split()
            self._build_elasticsearch(docs, force_rebuild)
        else:
            raise ValueError(f"不支持的存储类型: {VECTORSTORE_TYPE}")

    def similarity_search(
        self, query: str, top_k: int = 5
    ) -> List[Document]:
        """相似度检索"""
        if not self.vectorstore:
            self.build_vectorstore()

        return self.vectorstore.similarity_search(
            query=query,
            k=top_k,
        )

    def similarity_search_with_score(
        self, query: str, top_k: int = 5
    ) -> List[tuple]:
        """带分数的相似度检索"""
        if not self.vectorstore:
            self.build_vectorstore()

        return self.vectorstore.similarity_search_with_score(
            query=query,
            k=top_k,
        )

    def get_retriever(self, top_k: int = 5):
        """获取检索器"""
        if not self.vectorstore:
            self.build_vectorstore()

        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k},
        )


def init_rag(data_dir: str = None, force_rebuild: bool = False) -> TravelRAG:
    """初始化RAG系统"""
    if data_dir is None:
        data_dir = str(Path(__file__).parent.parent / "data" / "knowledge")
    rag = TravelRAG(data_dir=data_dir)
    rag.build_vectorstore(force_rebuild=force_rebuild)
    return rag


if __name__ == "__main__":
    # 测试
    data_dir = str(Path(__file__).parent.parent / "data" / "travel_data")
    rag = TravelRAG(data_dir)

    # 测试检索
    results = rag.similarity_search("东京旅游推荐", top_k=3)
    for i, doc in enumerate(results):
        print(f"\n--- 结果 {i+1} ---")
        print(doc.page_content[:200])