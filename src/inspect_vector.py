#!/usr/bin/env python3
"""
查看向量库内容
用法: python src/inspect_vector.py
"""
import chromadb

client = chromadb.PersistentClient(path='data/vectorstore')
collection = client.get_collection('travel_guide')

print('=' * 50)
print('向量库信息')
print('=' * 50)
print(f'Collection名: travel_guide')
print(f'总文本块数: {collection.count()}')
print(f'向量维度: 384 (BAAI/bge-small-zh-v1.5)')
print()

# 查看前5条
peek = collection.peek(5)

print('=' * 50)
print('前5条文本块')
print('=' * 50)
for i, (doc_id, doc, meta) in enumerate(zip(peek['ids'], peek['documents'], peek['metadatas'])):
    print(f'\n--- 文本块 {i+1} ---')
    print(f'ID: {doc_id}')
    print(f'来源: {meta.get("source", "未知")}')
    print(f'内容: {doc[:150]}...')