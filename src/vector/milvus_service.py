from typing import List, Dict, Any, Optional

from src.client.milvus_client import get_milvus_client
from src.vector.embedding import embedding_by_baai
from src.vector.milvus_search_type import CollectionType


def insert(collection_type: CollectionType, data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    写入一条数据
    """
    mc = get_milvus_client()
    result = mc.insert(collection_name=collection_type.value[0], data=data)
    return result


def search(collection_type: CollectionType,
        query: List[str],
        top_k: int = 5,
        filter_expr: Optional[str] = None,
           ):
    """
    批量查询
    """
    mc = get_milvus_client()
    vecs = embedding_by_baai(query)
    result = mc.search(collection_name=collection_type.value[0],
                       query_vectors=vecs,
                       top_k=top_k,
                       vector_field=collection_type.value[1],
                       output_fields=collection_type.value[2],
                       filter_expr=filter_expr,
                       )
    return result