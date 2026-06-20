"""
LLM 工厂
统一入口,业务代码只需调 LLMFactory.from_config(cfg).chat(...)
不关心底层是哪个后端
"""
from copy import deepcopy
from typing import List, Dict, Optional
from .base import BaseLLMAdapter, LLMResponse
from .openai_compat_adapter import OpenAICompatAdapter
from .deepseek_adapter import DeepSeekAdapter
from .modelscope_adapter import ModelScopeAdapter
from .mock_adapter import MockAdapter
from .cache import LLMCache


_ADAPTER_REGISTRY = {
    "openai_compat": OpenAICompatAdapter,
    "deepseek": DeepSeekAdapter,
    "modelscope": ModelScopeAdapter,
    "mock": MockAdapter,
}


class LLMFactory:
    """
    LLM 调用统一入口
    使用方式:
        llm = LLMFactory.from_config(cfg)
        resp = llm.chat([{"role":"user","content":"hello"}])
    """
    
    def __init__(self, adapter: BaseLLMAdapter, cache: Optional[LLMCache] = None,
                 default_temperature: float = 0.3, default_max_tokens: int = 1024):
        self.adapter = adapter
        self.cache = cache
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        # 统计
        self.call_count = 0
        self.cache_hit = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
    
    @classmethod
    def from_config(cls, cfg: dict, backend_override: Optional[str] = None) -> "LLMFactory":
        """
        从完整配置构建
        Args:
            cfg: 完整配置(load_config 返回的 dict)
            backend_override: 显式指定后端名,优先级高于配置中的 default_backend
        """
        llm_cfg = cfg.get("llm", {})
        backend_name = backend_override or llm_cfg.get("default_backend", "mock")
        
        if backend_name not in _ADAPTER_REGISTRY:
            raise ValueError(
                f"Unknown backend: {backend_name}. "
                f"Available: {list(_ADAPTER_REGISTRY.keys())}"
            )
        
        backend_cfg = deepcopy(llm_cfg.get("backends", {}).get(backend_name, {}))
        adapter = _ADAPTER_REGISTRY[backend_name](backend_cfg)
        
        # 缓存
        cache_cfg = llm_cfg.get("cache", {})
        cache = LLMCache(
            cache_dir=cache_cfg.get("cache_dir", "./cache/llm_cache"),
            enable=cache_cfg.get("enable", True),
        )
        
        return cls(
            adapter=adapter,
            cache=cache,
            default_temperature=llm_cfg.get("temperature", 0.3),
            default_max_tokens=llm_cfg.get("max_tokens", 1024),
        )
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[dict] = None,
        use_cache: bool = True,
        **kwargs
    ) -> LLMResponse:
        """统一对话接口"""
        if temperature is None:
            temperature = self.default_temperature
        if max_tokens is None:
            max_tokens = self.default_max_tokens
        
        # 缓存查询
        cache_key = None
        if use_cache and self.cache and self.cache.enable:
            cache_key = LLMCache.make_key(
                self.adapter.name, self.adapter.model,
                messages, temperature, max_tokens, response_format
            )
            cached = self.cache.get(cache_key)
            if cached is not None:
                self.cache_hit += 1
                cached.cached = True
                return cached
        
        # 真实调用
        resp = self.adapter.chat(
            messages, temperature=temperature, max_tokens=max_tokens,
            response_format=response_format, **kwargs
        )
        
        # 写缓存
        if cache_key is not None:
            self.cache.set(cache_key, resp)
        
        self.call_count += 1
        self.total_prompt_tokens += resp.prompt_tokens
        self.total_completion_tokens += resp.completion_tokens
        return resp
    
    def stats(self) -> dict:
        """返回累计调用统计"""
        return {
            "backend": self.adapter.name,
            "model": self.adapter.model,
            "call_count": self.call_count,
            "cache_hit": self.cache_hit,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
        }
