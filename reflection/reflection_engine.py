"""
反思生成引擎
- 调 LLM 生成结构化反思
- 严格 JSON 解析与字段校验
- 失败时降级,避免推荐链路中断
"""
import json
import re
from typing import Dict, List, Optional, Any
import numpy as np
from llm_adapter import LLMFactory
from .experience_unit import ExperienceUnit, LESSON_TAGS
from utils.logger import get_logger

logger = get_logger("reflection.engine")


REFLECTION_SYSTEM_PROMPT = """\
你是一位资深的教育推荐策略分析师。
任务: 分析一次"推荐-学生反应"案例,产出可被复用的策略修正建议。

严格按 JSON 格式输出,不要其他任何内容:
{
  "lesson": "30~80字,自然语言总结反思核心结论",
  "tags": ["从下列封闭集合中选最多3个"],
  "action_delta": {
    "difficulty_shift": -2 到 +2 的整数,
    "prereq_check": true/false,
    "diversity_target": 0~1 浮点,
    "kc_focus_change": ["建议改换的KC,可为空"]
  },
  "self_confidence": 0~1 的浮点,
  "contradicts_prior": true/false (本反思是否与历史经验矛盾)
}

tags 封闭集合(只能从中选):
- overestimate_difficulty       推荐题目过难
- underestimate_difficulty      推荐过简单
- ignore_prerequisite           忽视前置知识点
- low_diversity                 多样性不足
- preference_mismatch           与偏好不匹配
- weak_kc_misidentified         薄弱KC识别错
- redundant_recommendation      重复推荐
- other                         其他
"""


class ReflectionEngine:
    """反思生成器(LLM驱动 + 严格校验)"""
    
    def __init__(self, llm: LLMFactory):
        self.llm = llm
    
    # ============ 主接口 ============
    def reflect(
        self,
        student_profile,
        recommendation: dict,
        simulated_results: dict,
        severity: str,
        applied_experiences: List[ExperienceUnit],
        round_id: int = 0,
    ) -> Optional[ExperienceUnit]:
        """
        生成一条反思经验
        Returns:
            新生成的 ExperienceUnit, 或 None(LLM 解析失败)
        """
        prompt = self._build_prompt(
            student_profile, recommendation, simulated_results,
            severity, applied_experiences
        )
        
        try:
            resp = self.llm.chat(
                messages=[
                    {"role": "system", "content": REFLECTION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            logger.warning(f"LLM reflect call failed: {e}")
            return None
        
        parsed = self._parse_and_validate(resp.content)
        if parsed is None:
            logger.warning(f"Failed to parse reflection JSON, content snippet: {resp.content[:200]}")
            return None
        
        # 构建 ExperienceUnit
        exp = ExperienceUnit(
            student_signature=student_profile.signature_dict,
            student_sig_embedding=student_profile.sig_embedding,
            action=self._extract_action(recommendation),
            outcome=self._extract_outcome(recommendation, simulated_results),
            lesson=parsed["lesson"],
            lesson_tags=parsed["tags"],
            suggested_action_delta=parsed["action_delta"],
            confidence=parsed["self_confidence"],
            severity=severity,
            source_student_id=getattr(student_profile, "student_id", ""),
            source_round=round_id,
        )
        return exp
    
    # ============ Prompt 构造 ============
    def _build_prompt(
        self,
        student_profile,
        recommendation: dict,
        simulated_results: dict,
        severity: str,
        applied_experiences: List[ExperienceUnit],
    ) -> str:
        if applied_experiences:
            applied_summary = "\n".join([
                f"- [{e.exp_id}] {e.lesson} (置信度{e.confidence:.2f})"
                for e in applied_experiences
            ])
        else:
            applied_summary = "(本次推荐未使用任何历史经验)"
        
        # 预测 vs 实际表
        probs = recommendation.get("predicted_correct_rates", [])
        sims = simulated_results.get("simulated_correct", [])
        n = min(len(probs), len(sims))
        rows = ["| 题号 | 预测正确率 | 实际(模拟) |", "|---|---|---|"]
        for i in range(n):
            rows.append(f"| Q{i+1} | {probs[i]:.2f} | {sims[i]} |")
        pred_table = "\n".join(rows)
        
        # 学习增益
        expected_kg = recommendation.get("expected_kg", 0.0)
        actual_kg = simulated_results.get("actual_kg", 0.0)
        
        # 推荐动作
        rec_brief = (
            f"- KC焦点: {recommendation.get('kc_focus', [])}\n"
            f"- 难度分布: {recommendation.get('difficulty_distribution', {})}\n"
            f"- 策略标签: {recommendation.get('strategy_label', '')}\n"
            f"- 推荐理由: {recommendation.get('rationale', '')[:200]}"
        )
        
        return f"""\
## 案例严重度
{severity}

## 学生画像
- 薄弱知识点: {student_profile.weak_kcs}
- 能力水平: {student_profile.ability_level} (平均掌握度 {student_profile.avg_mastery:.2f})
- 学习偏好: {student_profile.learning_preference}
- 历史正确率: {student_profile.correct_rate:.2f}
- 交互密度: {student_profile.interaction_density}

## 本次推荐使用的历史经验
{applied_summary}

## 本次推荐动作
{rec_brief}

## 推荐器预测 vs 学生模拟结果
{pred_table}

## 学习增益
- 预期: {expected_kg:.3f}
- 实际: {actual_kg:.3f}
- 偏差: {expected_kg - actual_kg:.3f}

## 任务
请分析:
1. 此次推荐策略的核心问题是什么?
2. 如果再次遇到类似画像的学生,应该如何调整?
3. 已使用的历史经验是否帮上忙? 是否有需要反驳的?

严格按 system 提示的 JSON 格式输出。
"""
    
    # ============ JSON 解析 + 校验 ============
    @staticmethod
    def _parse_and_validate(content: str) -> Optional[Dict[str, Any]]:
        from utils import extract_json
        data = extract_json(content)
        if data is None or not isinstance(data, dict):
            return None
        
        # 字段校验 + 默认值
        try:
            lesson = str(data.get("lesson", "")).strip()
            if not lesson or len(lesson) > 200:
                lesson = lesson[:200] if lesson else "(LLM未给出明确反思)"
            
            tags = data.get("tags", [])
            if not isinstance(tags, list):
                tags = []
            tags = [t for t in tags if t in LESSON_TAGS][:3]
            if not tags:
                tags = ["other"]
            
            ad_raw = data.get("action_delta", {}) or {}
            if not isinstance(ad_raw, dict):
                ad_raw = {}
            action_delta = {
                "difficulty_shift": int(np.clip(int(ad_raw.get("difficulty_shift", 0) or 0), -2, 2)),
                "prereq_check": bool(ad_raw.get("prereq_check", False)),
                "diversity_target": float(np.clip(float(ad_raw.get("diversity_target", 0.4) or 0.4), 0.0, 1.0)),
                "kc_focus_change": (
                    ad_raw.get("kc_focus_change", []) 
                    if isinstance(ad_raw.get("kc_focus_change", []), list) else []
                ),
            }
            
            self_conf = float(np.clip(
                float(data.get("self_confidence", 0.5) or 0.5),
                0.0, 1.0
            ))
            
            return {
                "lesson": lesson,
                "tags": tags,
                "action_delta": action_delta,
                "self_confidence": self_conf,
                "contradicts_prior": bool(data.get("contradicts_prior", False)),
            }
        except (TypeError, ValueError) as e:
            logger.warning(f"validate_failed: {e}")
            return None
    
    # ============ 信息抽取 ============
    @staticmethod
    def _extract_action(rec: dict) -> dict:
        return {
            "kc_focus": rec.get("kc_focus", []),
            "difficulty_distribution": rec.get("difficulty_distribution", {}),
            "diversity_score": rec.get("diversity_score", 0.0),
            "strategy_label": rec.get("strategy_label", ""),
            "applied_experience_ids": rec.get("applied_experience_ids", []),
        }
    
    @staticmethod
    def _extract_outcome(rec: dict, sim: dict) -> dict:
        probs = rec.get("predicted_correct_rates", [])
        sims = sim.get("simulated_correct", [])
        n = min(len(probs), len(sims))
        mae = (
            float(np.mean([abs(probs[i] - sims[i]) for i in range(n)]))
            if n > 0 else 0.0
        )
        return {
            "predicted_correct_rates": list(probs),
            "simulated_correct": list(sims),
            "prediction_mae": mae,
            "expected_kg": rec.get("expected_kg", 0.0),
            "actual_kg": sim.get("actual_kg", 0.0),
        }
