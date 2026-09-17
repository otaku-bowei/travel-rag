"""
Milvus 客户端（最简版：只 insert + search）

保留方法：
  __init__ / close / insert / search
"""

import os
from typing import Optional, List, Dict, Any, Literal

from pymilvus import MilvusClient as PyMilvusClient

# Milvus 支持的 4 个一致性级别
ConsistencyLevel = Literal["Strong", "Session", "Bounded", "Eventually"]


class MilvusClient:
    def __init__(self, host: str = None, port: int = None,
                 user: str = None, password: str = None,
                 db_name: str = "default",
                 timeout: Optional[float] = None):
        self.host = host or os.getenv("MILVUS_HOST", "localhost")
        self.port = port or int(os.getenv("MILVUS_PORT", "19530"))
        self.user = user or os.getenv("MILVUS_USER", "")
        self.password = password or os.getenv("MILVUS_PASSWORD", "")
        self.db_name = db_name
        self.timeout = timeout

        uri = f"http://{self.host}:{self.port}"
        conn_kwargs = {"uri": uri, "db_name": db_name}
        if self.user:
            conn_kwargs["user"] = self.user
            conn_kwargs["password"] = self.password
        if self.timeout is not None:
            conn_kwargs["timeout"] = self.timeout

        self._mc = PyMilvusClient(**conn_kwargs)
        print(f"✅ Milvus 已连接: {self.host}:{self.port} (db={db_name})")

    def insert(self, collection_name: str, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """插入数据"""
        result = self._mc.insert(collection_name=collection_name, data=data)
        print(f"✅ 插入 {result['insert_count']} 条到 {collection_name}")
        return result

    def create_collection(
        self,
        collection_name: str,
        dim: int = 768,
        auto_id: bool = False,
        extra_fields: Optional[list] = None,
        description: Optional[str] = None,
        index_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """创建 collection（向量字段 + 自定义标量字段）

        Args:
            collection_name: collection 名
            dim: 向量维度（默认 768 = bge-base-zh-v1.5）
            auto_id: 是否让 Milvus 自动生成主键 id（默认 False，配合雪花 id）
            extra_fields: 额外标量字段列表 [FieldSchema, ...]，主键字段放最前
            description: collection 描述
            index_params: 索引参数（默认 AUTOINDEX + COSINE）
        """
        from pymilvus import FieldSchema, DataType, CollectionSchema

        # 默认必有向量字段
        fields = [FieldSchema("vector", DataType.FLOAT_VECTOR, dim=dim)]

        # 把用户的标量字段插到向量字段前面（主键字段放最前是 pymilvus 的约束）
        if extra_fields:
            fields = list(extra_fields) + fields

        schema = CollectionSchema(
            fields=fields,
            description=description,
            auto_id=auto_id,
        )

        if index_params is None:
            index_params = {
                "metric_type": "COSINE",
                "index_type": "AUTOINDEX",
            }

        self._mc.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
        )
        print(f"✅ {collection_name} 已创建（dim={dim}, auto_id={auto_id}）")

    def drop_collection(self, collection_name: str) -> None:
        """删除 collection"""
        self._mc.drop_collection(collection_name=collection_name)
        print(f"🗑️  {collection_name} 已删除")

    def has_collection(self, collection_name: str) -> bool:
        """判断 collection 是否存在"""
        return self._mc.has_collection(collection_name=collection_name)

    def search(
        self,
        collection_name: str,
        query_vectors: List[List[float]],
        top_k: int = 5,
        vector_field: str = "vector",
        output_fields: Optional[List[str]] = None,
        filter_expr: Optional[str] = None,
        metric_type: str = "COSINE",
        search_params: Optional[Dict[str, Any]] = None,
        consistency_level: ConsistencyLevel = "Strong",
    ) -> List[List[Dict[str, Any]]]:
        """
        向量检索

        Args:
            consistency_level: 一致性级别
                - "Strong": 强一致，立即看到最新写入（默认，测试推荐）
                - "Session": 同 session 立即可见
                - "Bounded": 最终一致（Milvus 默认，性能好）
                - "Eventually": 最弱一致，性能最好
        """
        kwargs: Dict[str, Any] = {
            "collection_name": collection_name,
            "data": query_vectors,
            "anns_field": vector_field,
            "limit": top_k,
            "output_fields": output_fields or [],
            "consistency_level": consistency_level,
        }
        if filter_expr:
            kwargs["filter"] = filter_expr

        sp: Dict[str, Any]
        if search_params is None:
            sp = {"metric_type": metric_type}
        else:
            sp = dict(search_params)
            sp.setdefault("metric_type", metric_type)
        kwargs["search_params"] = sp

        results = self._mc.search(**kwargs)

        all_results = []
        for hits in results:
            hits_list = []
            for hit in hits:
                item = {"id": hit["id"], "distance": hit["distance"]}
                for f in (output_fields or []):
                    item[f] = hit.get("entity", {}).get(f)
                hits_list.append(item)
            all_results.append(hits_list)
        return all_results

    def close(self) -> None:
        self._mc.close()
        print(f"🔌 Milvus 连接已关闭")


# ==================== 单例工厂 ====================

_milvus_instance: Optional[MilvusClient] = None


def get_milvus_client(host: str = None, port: int = None,
                      user: str = None, password: str = None,
                      db_name: str = "default",
                      force_new: bool = False) -> MilvusClient:
    """获取 Milvus 客户端实例（单例）"""
    global _milvus_instance
    if force_new or _milvus_instance is None:
        _milvus_instance = MilvusClient(
            host=host, port=port, user=user, password=password, db_name=db_name
        )
    return _milvus_instance


def reset_milvus_client() -> None:
    """重置单例（测试用）"""
    global _milvus_instance
    _milvus_instance = None
