"""
策略聚合器
将多条反思的 action_delta 聚合为本次推荐的策略修正
"""
import numpy as np
from typing import List, Dict
from reflection import ExperienceUnit


class StrategyAggregator:
    """聚合多个反思的 action_delta"""
    
    @staticmethod
    def aggregate(
        experiences: List[ExperienceUnit],
        use_structured_delta: bool = True,
    ) -> Dict:
        """
        Returns:
            {
              "difficulty_shift": int,
              "prereq_check": bool,
              "diversity_target": float,
              "kc_focus_change": List[str],
              "used_exp_ids": [...],
            }
        """
        if not experiences or not use_structured_delta:
            return {
                "difficulty_shift": 0,
                "prereq_check": False,
                "diversity_target": None,
                "kc_focus_change": [],
                "used_exp_ids": [e.exp_id for e in experiences],
            }
        
        weights = np.array([e.confidence for e in experiences], dtype=np.float32)
        if weights.sum() < 1e-6:
            weights = np.ones_like(weights) / len(weights)
        else:
            weights = weights / weights.sum()
        
        # difficulty_shift: 加权平均后四舍五入
        shifts = np.array([
            float(e.suggested_action_delta.get("difficulty_shift", 0))
            for e in experiences
        ])
        diff_shift = int(round(float(np.sum(shifts * weights))))
        diff_shift = max(-2, min(2, diff_shift))
        
        # prereq_check: 加权"投票" >= 0.5 即为 True
        prereq_votes = np.array([
            float(e.suggested_action_delta.get("prereq_check", False))
            for e in experiences
        ])
        prereq_score = float(np.sum(prereq_votes * weights))
        prereq_check = prereq_score >= 0.5
        
        # diversity_target: 加权平均
        div_targets = []
        div_weights = []
        for e, w in zip(experiences, weights):
            v = e.suggested_action_delta.get("diversity_target", None)
            if v is not None:
                div_targets.append(float(v))
                div_weights.append(w)
        diversity_target = (
            float(np.average(div_targets, weights=div_weights))
            if div_targets else None
        )
        
        # kc_focus_change: 收集所有提议(去重,保留出现次数)
        kc_changes = []
        for e in experiences:
            for kc in (e.suggested_action_delta.get("kc_focus_change") or []):
                if kc and kc not in kc_changes:
                    kc_changes.append(kc)
        
        return {
            "difficulty_shift": diff_shift,
            "prereq_check": prereq_check,
            "diversity_target": diversity_target,
            "kc_focus_change": kc_changes,
            "used_exp_ids": [e.exp_id for e in experiences],
        }
    
    @staticmethod
    def lessons_text(experiences: List[ExperienceUnit], min_confidence: float = 0.5) -> str:
        """供 L3 LLM 的自然语言反思集"""
        items = [
            f"- {e.lesson}"
            for e in experiences if e.confidence >= min_confidence
        ]
        return "\n".join(items)
