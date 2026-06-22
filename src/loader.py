"""
文档加载模块 - 支持多种格式的旅游攻略文档加载
"""
import os
from pathlib import Path
from typing import List
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    BSHTMLLoader,
)
from langchain_core.documents import Document


# 支持的文件格式和对应加载器
LOADERS = {
    ".txt": TextLoader,
    ".md": TextLoader,
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".html": BSHTMLLoader,
}


class TravelDocLoader:
    """旅游文档加载器"""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    def load_directory(self) -> List[Document]:
        """加载目录下所有支持的文档"""
        documents = []

        for file_path in self.data_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in LOADERS:
                try:
                    loader_class = LOADERS[file_path.suffix.lower()]
                    loader = loader_class(str(file_path), encoding="utf-8")
                    docs = loader.load()

                    # 添加来源元数据
                    for doc in docs:
                        doc.metadata["source"] = str(file_path.relative_to(self.data_dir))
                        doc.metadata["filename"] = file_path.name

                    documents.extend(docs)
                    print(f"✅ 加载成功: {file_path.name}")
                except Exception as e:
                    print(f"❌ 加载失败: {file_path.name} - {e}")

        return documents

    def load_file(self, file_path: str) -> List[Document]:
        """加载单个文件"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        suffix = path.suffix.lower()
        if suffix not in LOADERS:
            raise ValueError(f"不支持的文件格式: {suffix}")

        loader = LOADERS[suffix](str(path), encoding="utf-8")
        return loader.load()


def create_sample_data():
    """创建示例数据用于测试"""
    sample_dir = Path(__file__).parent.parent / "data" / "knowledge"
    sample_dir.mkdir(parents=True, exist_ok=True)

    tokyo_content = """# 东京5日游攻略

## 行程安排
### Day 1: 浅草寺 + 晴空塔
### Day 2: 涩谷 + 原宿
### Day 3: 迪士尼乐园

## 美食推荐
1. 筑地市场 - 寿司大
2. 一兰拉面

## 交通建议
- 推荐购买东京地铁券
"""

    sample_file = sample_dir / "japan_tokyo.md"
    if not sample_file.exists():
        sample_file.write_text(tokyo_content, encoding="utf-8")
        print(f"✅ 创建示例数据: {sample_file}")


if __name__ == "__main__":
    create_sample_data()
    loader = TravelDocLoader("data/knowledge")
    docs = loader.load_directory()
    print(f"\n共加载 {len(docs)} 个文档")