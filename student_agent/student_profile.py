"""学生画像数据结构"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np


@dataclass
class StudentProfile:
    """
    学生画像
    传给教师Agent的核心数据结构
    """
    student_id: str
    
    # ---- 来自DKT的诊断 ----
    kc_mastery: Dict[str, float] = field(default_factory=dict)   # KC → 掌握度[0,1]
    weak_kcs: List[str] = field(default_factory=list)            # 掌握度最低的前N个
    ability_level: str = "middle"                                # low/middle/high
    avg_mastery: float = 0.5
    
    # ---- 来自交互序列的统计 ----
    interaction_count: int = 0
    correct_rate: float = 0.5
    interaction_density: str = "medium"                          # sparse/medium/dense
    answered_qids: List[str] = field(default_factory=list)       # 已做过的题(避免重复推荐)
    
    # ---- 来自Profile Generator的自然语言摘要 ----
    weak_kcs_summary: str = ""
    ability_summary: str = ""
    preference_summary: str = ""
    learning_preference: str = "balanced"   # step_by_step/visual/balanced/...
    
    # ---- 表征向量 ----
    sig_embedding: Optional[np.ndarray] = None    # 签名嵌入,用于反思检索
    individual_embedding: Optional[np.ndarray] = None  # 个体表征
    group_embedding: Optional[np.ndarray] = None       # 群体先验
    fused_embedding: Optional[np.ndarray] = None       # 融合后的最终表征
    
    @property
    def signature_dict(self) -> dict:
        """供反思机制用的签名字典(便于人读和debug)"""
        return {
            "weak_kcs": self.weak_kcs[:5],
            "ability_level": self.ability_level,
            "learning_preference": self.learning_preference,
            "interaction_density": self.interaction_density,
        }
    
    def to_brief_text(self) -> str:
        """供LLM prompt用的简短描述"""
        return (
            f"学生ID: {self.student_id}\n"
            f"薄弱知识点: {', '.join(self.weak_kcs[:5])}\n"
            f"能力水平: {self.ability_level} (平均掌握度 {self.avg_mastery:.2f})\n"
            f"学习偏好: {self.learning_preference}\n"
            f"交互密度: {self.interaction_density} (共{self.interaction_count}次)\n"
            f"历史正确率: {self.correct_rate:.2f}"
        )
