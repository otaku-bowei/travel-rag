
'''
提问类型：
    景点
    路线
    时间规划
    交通
    天气
    美食
'''
from enum import Enum


class CollectionType(Enum):
    TRAVEL_DOCS = ("travel_docs", "vector", ["id", "content", "source", "category", "create_at", "update_at",])
    TRAVEL_DOCS_META = ("travel_docs_meta", "title_vector", ["id", "title", "likes", "line", "travel_doc_id", "create_at", "update_at",])
    TRAVEL_QUERY_CACHE = ("travel_query_cache", "query_vector", ["id", "query", "response", "create_at", "update_at",])
