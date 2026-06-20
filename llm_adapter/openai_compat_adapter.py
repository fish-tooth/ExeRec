"""
OpenAI 兼容协议后端
适用于:
- OpenAI 官方
- vLLM 本地部署(启动时加 --api-key 即兼容)
- 任何遵循 /v1/chat/completions 协议的服务
"""
import os
import time
import json
from typing import List, Dict, Optional
from .base import BaseLLMAdapter, LLMResponse


class OpenAICompatAdapter(BaseLLMAdapter):
    name = "openai_compat"
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.base_url = config.get("base_url", "").rstrip("/")
        api_key_env = config.get("api_key_env", "OPENAI_API_KEY")
        self.api_key = os.environ.get(api_key_env, "EMPTY")
        
        # 延迟导入 openai,避免硬依赖
        try:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
            )
        except ImportError as e:
            raise ImportError(
                "openai package required. Install: pip install openai>=1.0"
            ) from e
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        response_format: Optional[dict] = None,
        **kwargs
    ) -> LLMResponse:
        start = time.time()
        last_err = None
        
        # 重试逻辑
        for attempt in range(self.max_retries):
            try:
                req_kwargs = dict(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                if response_format is not None:
                    req_kwargs["response_format"] = response_format
                
                resp = self._client.chat.completions.create(**req_kwargs)
                latency_ms = (time.time() - start) * 1000
                
                usage = getattr(resp, "usage", None)
                pt = getattr(usage, "prompt_tokens", 0) if usage else 0
                ct = getattr(usage, "completion_tokens", 0) if usage else 0
                
                return LLMResponse(
                    content=resp.choices[0].message.content or "",
                    model=self.model,
                    backend=self.name,
                    prompt_tokens=pt,
                    completion_tokens=ct,
                    total_tokens=pt + ct,
                    latency_ms=latency_ms,
                    raw=resp,
                )
            except Exception as e:
                last_err = e
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                continue
        
        raise RuntimeError(
            f"OpenAI-compat call failed after {self.max_retries} retries: {last_err}"
        )
