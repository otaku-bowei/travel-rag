# src/cache/llm_cache.py

import redis
import json
import os
from typing import Optional


class RedisCache:
    def __init__(self, prefix : str, redis_url: str = "redis://localhost:6379"):
        self.redis = redis.from_url(redis_url)
        self.prefix = prefix

    def save_config(self, cache_key : str, config: dict, ttl: int = 3600):
        """缓存 LLM 配置（1h）"""
        self.redis.setex(
            self._deal_prefix(cache_key),
            ttl,
            json.dumps(config)
        )
        print(f"✅ LLM 配置已缓存: {self._deal_prefix(cache_key)}:{config}")

    def get_config(self, cache_key : str) -> Optional[dict]:
        """获取缓存的配置"""
        data = self.redis.get(self._deal_prefix(cache_key))
        if data:
            config = json.loads(data)
            return config
        return None

    def get_or_default_config(self, cache_key : str, func) -> dict:
        """获取缓存的配置"""
        data = self.redis.get(self._deal_prefix(cache_key))
        if data:
            config = json.loads(data)
            return config
        return func(cache_key)

    def delete_llm_config(self, cache_key):
        """删除缓存"""
        self.redis.delete(self._deal_prefix(cache_key))
        print(f"🗑️缓存{self._deal_prefix(cache_key)}已清除")


    def _deal_prefix(self, cache_key : str):
        return '[' + self.prefix + '_' + cache_key + ']'