#!/usr/bin/env python3
"""
旅游RAG系统 - 启动脚本
用法:
    python run.py              # 交互式对话
    python run.py "问题"   # 单次问答
"""
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.rag import TravelRAG
from src.chat import TravelChat
from src.config import MINIMAX_API_KEY, LLM_PROVIDER


def main():
    # 检查API Key
    if LLM_PROVIDER == "minimax" and not MINIMAX_API_KEY:
        print("⚠️ 请设置 MiniMax API Key:")
        print("   export MINIMAX_API_KEY=\"your-api-key\"")
        print()
        print("或者修改 .env 文件配置")
        return

    # 初始化RAG系统
    print("🔄 初始化RAG系统...")
    print("   知识库: data/knowledge/")
    print("   向量库: data/vectorstore/")
    print()

    rag = TravelRAG(
        data_dir=os.path.join(os.path.dirname(__file__), "data", "knowledge")
    )
    rag.build_vectorstore()

    # 初始化对话
    chat = TravelChat(rag)

    if len(sys.argv) > 1:
        # 单次问答
        question = " ".join(sys.argv[1:])
        print(f"❓ 问题: {question}")
        print("\n📝 回答:")
        result = chat.chat(question)
        print(result["answer"])
        print("\n📚 参考来源:")
        for i, src in enumerate(result["sources"], 1):
            print(f"   {i}. {src['source']}")
    else:
        # 交互式对话
        print("🎉 旅游RAG系统已启动!")
        print("   输入问题开始对话，输入 q 退出")
        print()
        while True:
            try:
                question = input("❓ ").strip()
                if not question:
                    continue
                if question.lower() in ["q", "quit", "exit"]:
                    print("👋 再见!")
                    break

                print("\n📝 回答:")
                result = chat.chat(question)
                print(result["answer"])
                print("\n📚 参考来源:")
                for i, src in enumerate(result["sources"], 1):
                    print(f"   {i}. {src['source'][:80]}...")
                print()
            except KeyboardInterrupt:
                print("\n👋 再见!")
                break
            except Exception as e:
                print(f"❌ 错误: {e}")


if __name__ == "__main__":
    main()