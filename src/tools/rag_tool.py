from src.vector.chroma_service import ChromaService
from langchain_core.tools import tool
from src.vector.milvus_service import search as milvus_search
from src.vector.milvus_search_type import CollectionType

@tool
def travel_rag_search(query:str) -> list[list[str]]:
    """从旅游知识库中检索景点、美食、攻略等信息。

    何时使用：
    - 用户问具体景点/餐厅/路线详情（"X 有什么好吃的"、"X 怎么玩"）
    - 用户问攻略、行程、推荐
    - 涉及京都、大阪等日本城市的旅游问题

    Args:
        query: 用户的查询字符串，要包含地点和关键词

    Returns:
        相关文档片段的拼接文本

    注意：
    - 比 LLM 训练数据更新更准确
    - 优先于 search（web 搜索）使用
    """
    # cs = ChromaService("../../data/chroma")
    # documents = cs.search(query)
    documents = milvus_search(CollectionType.TRAVEL_DOCS, query=[query])
    print(f"查询milvus做rag匹配：{documents}")
    ans : list[list[str]] = []
    for ld in documents:
        l : list[str] = []
        for d in ld:
            l.append(d.get("content"))
        ans.append(l)
    return ans
