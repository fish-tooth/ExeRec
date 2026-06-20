"""
学生画像生成器
根据 DKT 诊断结果 + 交互序列, 调用 LLM 生成自然语言画像
"""
import json
import re
from typing import Dict, List, Optional
from llm_adapter import LLMFactory
from utils.logger import get_logger

logger = get_logger("student.profile_gen")


PROFILE_SYSTEM_PROMPT = """\
你是一位资深的教育评估专家。
你的任务: 根据学生的知识点掌握情况和历史答题数据,产出三个维度的简短画像描述。

严格按 JSON 输出, 不要任何其他内容:
{
  "weak_kcs_summary": "30字以内,描述学生薄弱知识点的核心特征",
  "ability_summary": "30字以内,描述学生整体能力水平及特点",
  "preference_summary": "30字以内,描述学生学习偏好(题型/难度/解题方式倾向)",
  "learning_preference_tag": "选一个: step_by_step / visual / fast_drill / deep_analysis / balanced"
}
"""


class ProfileGenerator:
    """LLM 驱动的画像生成器"""
    
    def __init__(self, llm: LLMFactory):
        self.llm = llm
    
    def generate(
        self,
        student_id: str,
        weak_kcs: List[str],
        avg_mastery: float,
        correct_rate: float,
        interaction_count: int,
        recent_questions_brief: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """
        Args:
            recent_questions_brief: 最近几道题的简要描述(题型+难度+KC)
        
        Returns:
            {weak_kcs_summary, ability_summary, preference_summary, learning_preference_tag}
        """
        weak_text = ", ".join(weak_kcs[:5]) if weak_kcs else "无明显薄弱点"
        recent_text = "\n".join(recent_questions_brief[:5]) if recent_questions_brief else "无近期数据"
        
        user_prompt = (
            f"## 学生 {student_id} 的诊断数据\n"
            f"- 薄弱知识点(掌握度最低5项): {weak_text}\n"
            f"- 平均掌握度: {avg_mastery:.2f}\n"
            f"- 历史答题正确率: {correct_rate:.2f}\n"
            f"- 累计交互次数: {interaction_count}\n\n"
            f"## 最近答题简要\n{recent_text}\n\n"
            f"请按要求的 JSON 格式输出画像。"
        )
        
        messages = [
            {"role": "system", "content": PROFILE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        
        try:
            resp = self.llm.chat(
                messages,
                temperature=0.3,
                response_format={"type": "json_object"},
            )
            return self._parse(resp.content)
        except Exception as e:
            logger.warning(f"ProfileGenerator failed: {e}; using fallback")
            return self._fallback(weak_kcs, avg_mastery, correct_rate)
    
    @staticmethod
    def _parse(content: str) -> Dict[str, str]:
        """解析 LLM 返回的 JSON"""
        # 兼容 ```json 包裹
        content = content.strip()
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if m:
            content = m.group(0)
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return ProfileGenerator._fallback([], 0.5, 0.5)
        
        return {
            "weak_kcs_summary": str(data.get("weak_kcs_summary", "")).strip(),
            "ability_summary": str(data.get("ability_summary", "")).strip(),
            "preference_summary": str(data.get("preference_summary", "")).strip(),
            "learning_preference_tag": str(
                data.get("learning_preference_tag", "balanced")
            ).strip(),
        }
    
    @staticmethod
    def _fallback(weak_kcs, avg_mastery, correct_rate) -> Dict[str, str]:
        return {
            "weak_kcs_summary": f"薄弱于 {', '.join(weak_kcs[:3])}" if weak_kcs else "无明显薄弱",
            "ability_summary": (
                f"能力{'偏低' if avg_mastery < 0.4 else '中等' if avg_mastery < 0.7 else '较强'},"
                f"正确率约 {correct_rate:.2f}"
            ),
            "preference_summary": "数据不足以推断学习偏好",
            "learning_preference_tag": "balanced",
        }
