"""
反思触发判断
基于阈值的快速规则判定,决定是否生成反思 + 严重度
"""
from typing import List, Tuple
import numpy as np


class TriggerJudge:
    """决定是否触发反思 + 严重度等级"""
    
    def __init__(
        self,
        mae_high: float = 0.40,
        mae_medium: float = 0.25,
        mae_low: float = 0.15,
        kg_gap_high: float = 0.15,
        kg_gap_medium: float = 0.08,
    ):
        self.mae_high = mae_high
        self.mae_medium = mae_medium
        self.mae_low = mae_low
        self.kg_gap_high = kg_gap_high
        self.kg_gap_medium = kg_gap_medium
    
    def judge(
        self,
        predicted_probs: List[float],
        simulated_correct: List[int],
        expected_kg: float = 0.0,
        actual_kg: float = 0.0,
    ) -> Tuple[bool, str, dict]:
        """
        Returns:
            (should_reflect, severity, debug_dict)
            severity: 'high' / 'medium' / 'low' / 'none'
        """
        if not predicted_probs or not simulated_correct:
            return False, "none", {"reason": "empty_input"}
        
        # MAE
        mae = float(np.mean([abs(p - a) for p, a in zip(predicted_probs, simulated_correct)]))
        
        # KG gap (正值表示推荐方高估学习效果)
        kg_gap = expected_kg - actual_kg
        
        # 系统性偏差
        all_correct = sum(simulated_correct) == len(simulated_correct)
        all_wrong = sum(simulated_correct) == 0
        avg_pred = float(np.mean(predicted_probs))
        
        # 预测全 0.5 = 教师 Agent L3 走了 fallback,预测无信息
        # 此时 MAE 必然约 0.5,不该触发"高严重度反思"(浪费 LLM 调用)
        pred_array = np.asarray(predicted_probs, dtype=np.float64)
        pred_is_degenerate = bool(
            np.std(pred_array) < 1e-6 and abs(pred_array.mean() - 0.5) < 1e-3
        )
        
        debug = {
            "mae": mae,
            "kg_gap": kg_gap,
            "all_correct": all_correct,
            "all_wrong": all_wrong,
            "avg_pred": avg_pred,
            "pred_is_degenerate": pred_is_degenerate,
        }
        
        if pred_is_degenerate:
            # 预测无信息,无法用 MAE 判断推荐质量
            # 仍可用 kg_gap 判断,但不基于 MAE
            if kg_gap >= self.kg_gap_high:
                return True, "high", debug
            if kg_gap >= self.kg_gap_medium:
                return True, "medium", debug
            return False, "none", debug
        
        # ---- 高严重度 ----
        if (all_wrong and avg_pred > 0.5) or (all_correct and avg_pred < 0.5):
            return True, "high", debug
        if mae >= self.mae_high or kg_gap >= self.kg_gap_high:
            return True, "high", debug
        
        # ---- 中严重度 ----
        if mae >= self.mae_medium or kg_gap >= self.kg_gap_medium:
            return True, "medium", debug
        
        # ---- 低严重度 ----
        if mae >= self.mae_low:
            return True, "low", debug
        
        return False, "none", debug
