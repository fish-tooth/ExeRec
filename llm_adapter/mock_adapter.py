"""
Mock 后端
用于单元测试 和 离线调试
根据 prompt 中的特征关键词返回预制的结构化 JSON
"""
import json
import time
import re
from typing import List, Dict, Optional
from .base import BaseLLMAdapter, LLMResponse


class MockAdapter(BaseLLMAdapter):
    name = "mock"
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.model = "mock-llm"
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        response_format: Optional[dict] = None,
        **kwargs
    ) -> LLMResponse:
        start = time.time()
        time.sleep(0.01)  # 模拟少量延迟
        
        user_content = ""
        for m in messages:
            if m["role"] == "user":
                user_content += m["content"]
        
        # 根据 prompt 内容决定返回什么
        content = self._gen_mock_response(user_content, response_format)
        
        latency_ms = (time.time() - start) * 1000
        return LLMResponse(
            content=content,
            model=self.model,
            backend=self.name,
            prompt_tokens=len(user_content) // 4,
            completion_tokens=len(content) // 4,
            total_tokens=(len(user_content) + len(content)) // 4,
            latency_ms=latency_ms,
            cached=False,
        )
    
    def _gen_mock_response(self, prompt: str, response_format: Optional[dict]) -> str:
        """根据 prompt 关键词返回不同结构的 mock 数据"""
        want_json = (response_format and response_format.get("type") == "json_object")
        
        # 反思生成场景
        if "反思" in prompt or "lesson" in prompt or "action_delta" in prompt:
            mock = {
                "lesson": "本次推荐难度偏高,该类学生应优先巩固前置知识点",
                "tags": ["overestimate_difficulty", "ignore_prerequisite"],
                "action_delta": {
                    "difficulty_shift": -1,
                    "prereq_check": True,
                    "diversity_target": 0.5,
                    "kc_focus_change": []
                },
                "self_confidence": 0.7,
                "contradicts_prior": False
            }
            return json.dumps(mock, ensure_ascii=False)
        
        # 学生画像生成场景
        if "学生画像" in prompt or "profile" in prompt.lower():
            mock = {
                "weak_kcs_summary": "在回归直线方程和平均数计算上掌握薄弱",
                "ability_summary": "中等水平,具备基础运算能力但综合应用题易错",
                "preference_summary": "偏好分步推导,对图表型题目接受度较高"
            }
            return json.dumps(mock, ensure_ascii=False) if want_json else str(mock)
        
        # 推荐理由生成场景
        if "推荐理由" in prompt or "rationale" in prompt:
            mock = {
                "rationale": "针对学生薄弱的回归方程KC,选择中等难度题目巩固",
                "predicted_correct_rates": [0.7, 0.6, 0.55, 0.5, 0.45],
                "strategy_label": "weak_kc_first"
            }
            return json.dumps(mock, ensure_ascii=False)
        
        # 默认返回
        if want_json:
            return json.dumps({"response": "mock response"}, ensure_ascii=False)
        return "Mock response."
