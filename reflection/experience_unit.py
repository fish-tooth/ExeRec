"""
经验单元: 反思机制的原子单位
每次"推荐-模拟-比对"产生 0 或 1 个 ExperienceUnit
"""
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
import numpy as np
import time
import uuid


# 反思类型封闭集合(用于结构化分析)
LESSON_TAGS = [
    "overestimate_difficulty",      # 推荐过难
    "underestimate_difficulty",     # 推荐过简单
    "ignore_prerequisite",          # 忽视前置 KC
    "low_diversity",                # 推荐多样性不足
    "preference_mismatch",          # 与学习偏好不匹配
    "weak_kc_misidentified",        # 薄弱 KC 识别错误
    "redundant_recommendation",     # 重复推荐
    "other",
]


@dataclass
class ExperienceUnit:
    """单条经验单元"""
    # ---- 标识 ----
    exp_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    # ---- 触发场景 ----
    student_signature: Dict[str, Any] = field(default_factory=dict)
    # e.g. {"weak_kcs":[...], "ability_level":"middle", "learning_preference":"step_by_step", "interaction_density":"sparse"}
    
    student_sig_embedding: Optional[np.ndarray] = None
    
    # ---- 推荐动作 ----
    action: Dict[str, Any] = field(default_factory=dict)
    # e.g. {"kc_focus":[...], "difficulty_distribution":{"easy":0,"medium":2,"hard":3},
    #       "diversity_score":0.3, "strategy_label":"weak_kc_first"}
    
    # ---- 结果 ----
    outcome: Dict[str, Any] = field(default_factory=dict)
    # e.g. {"predicted_correct_rates":[...], "simulated_correct":[...],
    #       "prediction_mae":0.42, "expected_kg":0.15, "actual_kg":0.03}
    
    # ---- 反思内容 ----
    lesson: str = ""
    lesson_tags: List[str] = field(default_factory=list)
    suggested_action_delta: Dict[str, Any] = field(default_factory=dict)
    # e.g. {"difficulty_shift":-1, "prereq_check":True,
    #       "diversity_target":0.5, "kc_focus_change":[]}
    
    # ---- 元数据 ----
    confidence: float = 0.5
    support_count: int = 1
    contradict_count: int = 0
    last_used_at: float = field(default_factory=time.time)
    use_count: int = 0
    status: str = "active"  # active / dormant / retired
    
    severity: str = "medium"  # high / medium / low
    
    # ---- 来源元信息(用于审计与debug)----
    source_student_id: str = ""
    source_round: int = 0
    
    def to_serializable(self) -> dict:
        """转为可 pickle 的字典"""
        d = asdict(self)
        # numpy array 单独处理
        if isinstance(self.student_sig_embedding, np.ndarray):
            d["student_sig_embedding"] = self.student_sig_embedding.tolist()
        return d
    
    @classmethod
    def from_serializable(cls, d: dict) -> "ExperienceUnit":
        emb = d.get("student_sig_embedding")
        if isinstance(emb, list):
            d["student_sig_embedding"] = np.array(emb, dtype=np.float32)
        return cls(**d)
    
    def to_brief_text(self) -> str:
        """供 LLM prompt 用的简短描述"""
        sig = self.student_signature
        return (
            f"[{self.exp_id}] (置信度 {self.confidence:.2f})\n"
            f"  场景: {sig.get('ability_level','?')} 能力, "
            f"偏好 {sig.get('learning_preference','?')}, "
            f"薄弱 {sig.get('weak_kcs',[])[:3]}\n"
            f"  教训: {self.lesson}\n"
            f"  建议调整: {self.suggested_action_delta}"
        )
