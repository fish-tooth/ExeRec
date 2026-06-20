"""
LLM 调用缓存
- 基于 prompt + 参数的 hash 作为 key
- 用 diskcache 持久化(可选), 没装时降级为内存字典
"""
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Optional


class LLMCache:
    """简单的 LLM 调用缓存层"""
    
    def __init__(self, cache_dir: str, enable: bool = True):
        self.enable = enable
        if not enable:
            return
        
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        try:
            import diskcache
            self.cache = diskcache.Cache(cache_dir)
            self.backend = "diskcache"
        except ImportError:
            self.cache = {}
            self.backend = "memory"
    
    @staticmethod
    def make_key(
        backend: str,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        response_format: Optional[dict],
    ) -> str:
        payload = {
            "backend": backend,
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": response_format,
        }
        s = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()
    
    def get(self, key: str):
        if not self.enable:
            return None
        try:
            return self.cache.get(key) if hasattr(self.cache, "get") else self.cache.get(key, None)
        except Exception:
            return None
    
    def set(self, key: str, value):
        if not self.enable:
            return
        try:
            if hasattr(self.cache, "set"):
                self.cache.set(key, value)
            else:
                self.cache[key] = value
        except Exception:
            pass
