"""
ModelScope API-Inference 适配器
ModelScope 提供 OpenAI 兼容协议,直接复用
参考: https://www.modelscope.cn/docs/model-service/API-Inference/intro
"""
from .openai_compat_adapter import OpenAICompatAdapter


class ModelScopeAdapter(OpenAICompatAdapter):
    name = "modelscope"
    
    def __init__(self, config: dict):
        config.setdefault("base_url", "https://api-inference.modelscope.cn/v1")
        config.setdefault("model", "Qwen/Qwen2.5-72B-Instruct")
        config.setdefault("api_key_env", "MODELSCOPE_API_KEY")
        super().__init__(config)
