"""
对话模块 - LangChain RAG 对话入口
支持 RAG 未命中时直接请求 LLM
"""
import os
from pathlib import Path
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .config import (
    LLM_PROVIDER,
    MINIMAX_API_KEY,
    MINIMAX_BASE_URL,
    MINIMAX_MODEL,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    SYSTEM_PROMPT,
    TOP_K,
)
from .rag import TravelRAG
from .prompt_pool import build_fewshot_prompt, select_example


class TravelChat:
    """旅游问答对话系统"""

    def __init__(self, rag: TravelRAG):
        self.rag = rag
        self.llm = self._init_llm()

    def _init_llm(self):
        """初始化LLM"""
        if LLM_PROVIDER == "minimax":
            return ChatOpenAI(
                model=MINIMAX_MODEL,
                base_url=MINIMAX_BASE_URL,
                api_key=MINIMAX_API_KEY or "dummy",
                temperature=0.7,
                max_tokens=2000,
            )
        else:
            return ChatOpenAI(
                model=OPENAI_MODEL,
                api_key=OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", ""),
                temperature=0.7,
                max_tokens=2000,
            )

    def chat(self, question: str, use_fewshot: bool = True) -> dict:
        """对话 - 支持 RAG 未命中时 fallback"""
        # 0. Few-shot 增强
        enhanced_question = build_fewshot_prompt(question, use_fewshot)
        
        # 1. 尝试检索
        retriever = self.rag.get_retriever(top_k=TOP_K)
        docs = retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in docs])

        # 2. 判断是否检索成功
        if not docs or not context.strip():
            # RAG 未命中，直接请求 LLM
            print("⚠️ RAG 检索未命中，使用通用知识...")
            prompt = ChatPromptTemplate.from_messages([
                ("system", """你是一个专业的旅游顾问助手，熟悉全球热门旅游目的地的景点、美食、交通和行程规划。
请根据你的通用知识回答用户的问题。
注意：这是通用建议而非基于特定旅游攻略。"""),
                ("user", "用户问题：{question}\n\n请给出详细的旅游建议："),
            ])
            
            # Few-shot 增强
            if use_fewshot and select_example(question):
                # 重构 prompt 使用 Few-shot
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "你是一个专业的旅游顾问助手。请按以下示例格式回答。"),
                    ("user", "{question}\n\n请给出详细的旅游建议："),
                ])
        else:
            # RAG 命中，使用知识库
            print(f"✅ RAG 检索成功，找到 {len(docs)} 条相关内容")
            prompt = ChatPromptTemplate.from_messages([
                ("system", """你是一个专业的旅游顾问助手。
请根据提供的旅游攻略信息回答用户的问题。
如果无法从信息中找到答案，请如实告知。"""),
                ("system", "旅游攻略信息：\n{context}"),
                ("user", "用户问题：{question}\n\n请给出详细的旅游建议："),
            ])

        # 3. 调用 LLM
        chain = prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": enhanced_question})

        return {
            "answer": answer,
            "sources": [
                {
                    "content": doc.page_content[:300],
                    "source": doc.metadata.get("source", "未知"),
                }
                for doc in docs
            ],
            "rag_hit": bool(docs and context.strip()),
        }

    def stream_chat(self, question: str):
        """流式对话"""
        retriever = self.rag.get_retriever(top_k=TOP_K)
        docs = retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in docs])

        if not docs or not context.strip():
            prompt = ChatPromptTemplate.from_messages([
                ("system", """你是一个专业的旅游顾问助手。
注意：知识库检索未命中，请基于通用知识回答。"""),
                ("user", "用户问题：{question}"),
            ])
        else:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "旅游攻略信息：\n{context}"),
                ("user", "用户问题：{question}"),
            ])

        chain = prompt | self.llm | StrOutputParser()
        for chunk in chain.stream({"context": context, "question": question}):
            yield chunk


def create_chat(data_dir: str) -> TravelChat:
    """创建对话系统"""
    rag = TravelRAG(data_dir)
    rag.build_vectorstore()
    return TravelChat(rag)


if __name__ == "__main__":
    from .rag import TravelRAG
    data_dir = str(Path(__file__).parent.parent / "data" / "knowledge")
    chat = create_chat(data_dir)
    result = chat.chat("京都三日游怎么安排")
    print(result["answer"])