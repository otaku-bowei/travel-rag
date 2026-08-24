import re
from pathlib import Path
from typing import List

import chromadb
from chromadb import Settings, QueryResult
from chromadb.utils import embedding_functions
from langchain_chroma import Chroma
from langchain_core.documents import Document
from . import file_util as fu

class ChromaService:

    def __init__(self, persist_directory: str, collection_name="chroma"):
        embedding_function = embedding_functions.DefaultEmbeddingFunction()
        # self.chroma = Chroma(collection_name, embedding_function=embedding_function,
        #                      persist_directory=persist_directory)
        # 1. 创建 client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        # 2. 创建/获取 collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function
        )


    def save(self, text: str, metadata: dict = None):
        # self.chroma.add_texts(input)
        """保存单条文本"""
        doc_id = f"doc_{self.collection.count()}"
        self.collection.add(
            documents=[text],
            metadatas=[metadata or {}],
            ids=[doc_id]
        )
        return doc_id

    def search(self, query: str, n_results: int = 4) -> list[list[str]] | None:
        """
        默认近似搜索
        """
        # return self.chroma.search(input, search_type="similarity")
        """搜索，返回原始结果"""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results.get('documents')


    def import_md_file(self, file_path: str) -> int:
        """导入 md 文件到向量库
        Args:
            file_path: md 文件路径
        Returns:
            导入的段落数量
        """
        if not file_path.endswith(".md") :
            raise TypeError(f"该文件为非md文件: {file_path}")
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        content = path.read_text(encoding='utf-8')
        # 按段落切片，过滤空段落，并且清理文本
        paragraphs = fu.split_paragraphs(content)
        cleaned_paragraphs = [fu.clean_text(p) for p in paragraphs]
        valid_paragraphs = [p for p in cleaned_paragraphs if p.strip()]
        if not valid_paragraphs:
            print(f"警告: {file_path} 没有有效内容")
            return 0
        # 构建 Document 列表
        documents = [
            Document(
                page_content=para,
                metadata={
                    "source": str(path),
                    "filename": path.name,
                    "paragraph_idx": i,
                }
            )
            for i, para in enumerate(valid_paragraphs)
        ]
        # 添加到向量库
        for d in documents:
            self.save(d.page_content, d.metadata)
        return len(documents)

