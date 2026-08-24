import re
from pathlib import Path
from typing import List

from chromadb.utils import embedding_functions
from langchain_chroma import Chroma
from langchain_core.documents import Document
from file_util import *

class ChromaService:

    def __init__(self, persist_directory: str, collection_name="chroma"):
        embedding_function = embedding_functions.DefaultEmbeddingFunction()
        self.chroma = Chroma(collection_name, embedding_function=embedding_function,
                             persist_directory=persist_directory)

    def save(self, input: str):
        self.chroma.add_texts(input)

    def search(self, input: str) -> list[Document]:
        """
        默认近似搜索
        """
        return self.chroma.search(input, search_type="similarity")

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
        paragraphs = split_paragraphs(content)
        cleaned_paragraphs = [clean_text(p) for p in paragraphs]
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
        self.chroma.add_documents(documents)
        return len(documents)
