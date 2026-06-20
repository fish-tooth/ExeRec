"""
DeepSeek 平台适配器
DeepSeek 协议与 OpenAI 兼容,直接复用 OpenAICompatAdapter 的逻辑
仅 base_url 和默认 model 不同
"""
from .openai_compat_adapter import OpenAICompatAdapter


class DeepSeekAdapter(OpenAICompatAdapter):
    name = "deepseek"
    
    def __init__(self, config: dict):
        # 强制默认值,允许配置覆盖
        config.setdefault("base_url", "https://api.deepseek.com/v1")
        config.setdefault("model", "deepseek-chat")
        config.setdefault("api_key_env", "DEEPSEEK_API_KEY")
        super().__init__(config)
