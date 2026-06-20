"""
LLM Adapter 基类
所有具体后端都继承此类
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any


@dataclass
class LLMResponse:
    """统一的 LLM 响应封装"""
    content: str
    model: str
    backend: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    raw: Optional[Any] = None
    cached: bool = False
    extra: Dict = field(default_factory=dict)


class BaseLLMAdapter(ABC):
    """LLM 后端适配器基类"""
    
    name: str = "base"
    
    def __init__(self, config: dict):
        """
        Args:
            config: 单个后端的配置(从 cfg.llm.backends.<name> 取)
        """
        self.config = config
        self.model = config.get("model", "")
        self.timeout = config.get("timeout", 60)
        self.max_retries = config.get("max_retries", 3)
    
    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        response_format: Optional[dict] = None,
        **kwargs
    ) -> LLMResponse:
        """对话补全接口"""
        raise NotImplementedError
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """嵌入接口(可选实现,本项目嵌入已离线生成)"""
        raise NotImplementedError(f"{self.name} does not support embedding")
