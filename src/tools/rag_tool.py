from src.vector.chroma_service import ChromaService
from langchain_core.tools import tool

@tool
def rag_search(input:str) -> list[list[str]]:
    """当用户问题涉及京都或大阪的时候,必须使用rag检索"""
    cs = ChromaService("../../data/chroma")
    documents = cs.search(input)
    return documents
