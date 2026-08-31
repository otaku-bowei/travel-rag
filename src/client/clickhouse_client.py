"""
ClickHouse 客户端
纯粹的连接和基础操作
"""

import os
from typing import Optional

import clickhouse_connect


class ClickHouseClient:
    """ClickHouse 客户端封装"""

    def __init__(self, host: str = None, port: int = None, database: str = None,
                 username: str = None, password: str = None):
        """
        初始化 ClickHouse 客户端

        Args:
            host: CH 主机地址
            port: CH 端口，默认 8123
            database: 数据库名
            username: 用户名
            password: 密码
        """
        self.host = host or os.getenv("CH_HOST", "localhost")
        self.port = port or int(os.getenv("CH_PORT", "8123"))
        self.database = database or os.getenv("CH_DATABASE", "travel_rag")
        self.username = username or os.getenv("CH_USERNAME", "default")
        self.password = password or os.getenv("CH_PASSWORD", "")

        self.client = clickhouse_connect.get_client(
            host=self.host,
            port=self.port,
            database=self.database,
            username=self.username,
            password=self.password
        )
        print(f"✅ ClickHouse 已连接: {self.host}:{self.port}/{self.database}")

    def execute(self, sql: str) -> Any:
        """执行 SQL"""
        return self.client.command(sql)

    def query(self, sql: str) -> Any:
        """执行查询"""
        return self.client.query(sql).result_rows

    def insert(self, table: str, data: list):
        """插入数据"""
        self.client.insert(table, data)

    def close(self):
        """关闭连接"""
        self.client.close()


# 全局单例
_client: Optional[ClickHouseClient] = None


def get_client() -> ClickHouseClient:
    """获取全局客户端单例"""
    global _client
    if _client is None:
        _client = ClickHouseClient()
    return _client


def init_client(host: str = None, port: int = None, database: str = None,
                username: str = None, password: str = None) -> ClickHouseClient:
    """初始化客户端"""
    global _client
    _client = ClickHouseClient(host, port, database, username, password)
    return _client
