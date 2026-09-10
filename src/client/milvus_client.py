"""
Milvus 客户端
纯粹的连接和基础操作
"""

import os
from typing import Optional, List, Dict, Any

from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility


class MilvusClient:
    """Milvus 客户端封装"""

    def __init__(self, host: str = None, port: int = None,
                 user: str = None, password: str = None,
                 alias: str = "default"):
        """
        初始化 Milvus 客户端

        Args:
            host: Milvus 主机地址
            port: Milvus gRPC 端口，默认 19530
            user: 用户名
            password: 密码
            alias: 连接别名，默认 "default"
        """
        self.host = host or os.getenv("MILVUS_HOST", "localhost")
        self.port = port or int(os.getenv("MILVUS_PORT", "19530"))
        self.user = user or os.getenv("MILVUS_USER", "")
        self.password = password or os.getenv("MILVUS_PASSWORD", "")
        self.alias = alias

        # 连接 Milvus
        conn_params = {
            "alias": self.alias,
            "host": self.host,
            "port": str(self.port),
        }
        if self.user:
            conn_params["user"] = self.user
            conn_params["password"] = self.password

        connections.connect(**conn_params)
        print(f"✅ Milvus 已连接: {self.host}:{self.port} (alias={self.alias})")

    # ==================== Collection 管理 ====================

    def has_collection(self, collection_name: str) -> bool:
        """检查 collection 是否存在"""
        return utility.has_collection(collection_name, using=self.alias)

    def create_collection(self, collection_name: str, dim: int,
                          description: str = "",
                          primary_field: str = "id",
                          vector_field: str = "vector",
                          text_field: str = "text",
                          metric_type: str = "COSINE") -> Collection:
        """
        创建 collection

        Args:
            collection_name: collection 名
            dim: 向量维度
            description: 描述
            primary_field: 主键字段名
            vector_field: 向量字段名
            text_field: 文本字段名
            metric_type: 距离度量 (COSINE/L2/IP)

        Returns:
            Collection 对象
        """
        if self.has_collection(collection_name):
            print(f"⚠️  Collection 已存在: {collection_name}")
            return Collection(collection_name, using=self.alias)

        fields = [
            FieldSchema(name=primary_field, dtype=DataType.INT64,
                        is_primary=True, auto_id=True),
            FieldSchema(name=text_field, dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name=vector_field, dtype=DataType.FLOAT_VECTOR, dim=dim),
        ]
        schema = CollectionSchema(fields=fields, description=description)
        collection = Collection(
            name=collection_name,
            schema=schema,
            using=self.alias
        )
        print(f"✅ Collection 已创建: {collection_name} (dim={dim}, metric={metric_type})")
        return collection

    def get_collection(self, collection_name: str) -> Collection:
        """获取 collection"""
        if not self.has_collection(collection_name):
            raise ValueError(f"Collection 不存在: {collection_name}")
        return Collection(collection_name, using=self.alias)

    def drop_collection(self, collection_name: str) -> None:
        """删除 collection"""
        if self.has_collection(collection_name):
            utility.drop_collection(collection_name, using=self.alias)
            print(f"🗑️  Collection 已删除: {collection_name}")

    def list_collections(self) -> List[str]:
        """列出所有 collection"""
        return utility.list_collections(using=self.alias)

    # ==================== 索引管理 ====================

    def create_index(self, collection_name: str,
                     vector_field: str = "vector",
                     index_type: str = "IVF_FLAT",
                     metric_type: str = "COSINE",
                     params: Optional[Dict[str, Any]] = None) -> None:
        """
        创建向量索引

        Args:
            collection_name: collection 名
            vector_field: 向量字段
            index_type: 索引类型 (IVF_FLAT/HNSW/FLAT/AUTOINDEX)
            metric_type: 距离度量
            params: 索引参数，如 {"nlist": 128}
        """
        collection = self.get_collection(collection_name)
        index_params = {
            "metric_type": metric_type,
            "index_type": index_type,
            "params": params or {"nlist": 128},
        }
        collection.create_index(vector_field, index_params)
        print(f"✅ 索引已创建: {collection_name}.{vector_field} ({index_type})")

    def load_collection(self, collection_name: str) -> None:
        """加载 collection 到内存（查询前必须）"""
        collection = self.get_collection(collection_name)
        collection.load()
        print(f"✅ Collection 已加载: {collection_name}")

    def release_collection(self, collection_name: str) -> None:
        """释放 collection 从内存"""
        collection = self.get_collection(collection_name)
        collection.release()
        print(f"♻️  Collection 已释放: {collection_name}")

    # ==================== 数据操作 ====================

    def insert(self, collection_name: str, texts: List[str],
               vectors: List[List[float]]) -> Dict[str, Any]:
        """
        插入数据

        Args:
            collection_name: collection 名
            texts: 文本列表
            vectors: 向量列表（每个向量 dim 维）

        Returns:
            插入结果
        """
        assert len(texts) == len(vectors), "texts 和 vectors 长度必须一致"

        collection = self.get_collection(collection_name)
        data = [texts, vectors]  # 字段顺序：text, vector
        result = collection.insert(data)
        collection.flush()
        print(f"✅ 插入 {len(texts)} 条数据到 {collection_name}")
        return {
            "insert_count": result.insert_count,
            "primary_keys": result.primary_keys,
        }

    def search(self, collection_name: str, query_vectors: List[List[float]],
               top_k: int = 5,
               vector_field: str = "vector",
               output_fields: Optional[List[str]] = None,
               expr: Optional[str] = None) -> List[List[Dict[str, Any]]]:
        """
        向量检索

        Args:
            collection_name: collection 名
            query_vectors: 查询向量列表
            top_k: 返回 top-k 结果
            vector_field: 向量字段
            output_fields: 返回的额外字段
            expr: 过滤表达式，如 'id > 100'

        Returns:
            每条查询的 top-k 结果列表
        """
        collection = self.get_collection(collection_name)

        search_params = {"metric_type": "COSINE"}
        results = collection.search(
            data=query_vectors,
            anns_field=vector_field,
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=output_fields or ["text"],
        )

        # 格式化为易读结构
        all_results = []
        for hits in results:
            hits_list = []
            for hit in hits:
                item = {
                    "id": hit.id,
                    "distance": hit.distance,
                }
                if output_fields:
                    for f in output_fields:
                        item[f] = hit.entity.get(f)
                hits_list.append(item)
            all_results.append(hits_list)
        return all_results

    def count(self, collection_name: str, expr: Optional[str] = None) -> int:
        """统计行数"""
        collection = self.get_collection(collection_name)
        return collection.num_entities

    # ==================== 关闭 ====================

    def close(self) -> None:
        """关闭连接"""
        connections.disconnect(self.alias)
        print(f"🔌 Milvus 连接已关闭: {self.alias}")


# 便捷工厂
def get_milvus_client(host: str = None, port: int = None,
                      user: str = None, password: str = None) -> MilvusClient:
    """获取 Milvus 客户端实例（单例）"""
    global _milvus_instance
    try:
        if _milvus_instance is None:
            _milvus_instance = MilvusClient(host=host, port=port,
                                            user=user, password=password)
    except NameError:
        _milvus_instance = MilvusClient(host=host, port=port,
                                        user=user, password=password)
    return _milvus_instance


_milvus_instance = None
