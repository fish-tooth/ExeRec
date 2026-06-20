"""
全局 Agent
- 协调教师 Agent 与学生 Agent
- 持有 ExperienceBank 与 ReflectionEngine
- 负责"是否反思"判定 + 经验置信度回填
- 这是反思机制的中央调度

注意: 推荐过程实际由 TeacherAgent.recommend() 完成,
GlobalAgent 主要负责事后判断 + 反思生成 + 经验维护
"""
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import time

from reflection import ExperienceBank, ReflectionEngine, TriggerJudge, ExperienceUnit
from llm_adapter import LLMFactory
from utils.logger import get_logger

logger = get_logger("global.agent")


class GlobalAgent:
    """全局协调 + 反思维护"""
    
    def __init__(self, cfg: dict, llm: LLMFactory):
        self.cfg = cfg
        self.llm = llm
        
        ref_cfg = cfg.get("reflection", {})
        self.reflection_enabled = ref_cfg.get("enable", True)
        
        # ---- 经验库 ----
        conf_cfg = ref_cfg.get("confidence", {})
        self.exp_bank = ExperienceBank(
            retire_below=conf_cfg.get("retire_below", 0.20),
            dormant_below=conf_cfg.get("dormant_below", 0.35),
            decay_per_day=conf_cfg.get("decay_per_day", 0.99),
        )
        
        # 加载持久化经验库
        persist_path = ref_cfg.get("persist", {}).get("path", "./experience_bank_storage/exp_bank.pkl")
        self.persist_path = persist_path
        if Path(persist_path).exists():
            self.exp_bank.load(persist_path)
        
        self.autosave_interval = ref_cfg.get("persist", {}).get("autosave_interval", 50)
        
        # ---- 触发判断 ----
        trig_cfg = ref_cfg.get("triggers", {})
        self.judge = TriggerJudge(
            mae_high=trig_cfg.get("mae_high", 0.40),
            mae_medium=trig_cfg.get("mae_medium", 0.25),
            mae_low=trig_cfg.get("mae_low", 0.15),
            kg_gap_high=trig_cfg.get("kg_gap_high", 0.15),
            kg_gap_medium=trig_cfg.get("kg_gap_medium", 0.08),
        )
        
        # ---- 反思引擎 ----
        self.engine = ReflectionEngine(llm)
        
        # ---- 检索参数 ----
        ret_cfg = ref_cfg.get("retrieval", {})
        self.retrieval_top_k = ret_cfg.get("top_k", 3)
        self.retrieval_min_conf = ret_cfg.get("min_confidence", 0.4)
        self.retrieval_sim_thr = ret_cfg.get("similarity_threshold", 0.6)
        
        # ---- 合并 ----
        cons_cfg = ref_cfg.get("consolidation", {})
        self.consolidation_enabled = cons_cfg.get("enable", True)
        self.consolidation_interval = cons_cfg.get("interval", 100)
        self.consolidation_sim_thr = cons_cfg.get("similarity_threshold", 0.9)
        self.consolidation_min_cluster = cons_cfg.get("min_cluster_size", 3)
        self.consolidation_tag_thr = cons_cfg.get("tag_overlap_threshold", 0.6)
        
        # ---- 消融开关 ----
        abl = ref_cfg.get("ablation", {})
        self.abl_structured_delta = abl.get("structured_delta", True)
        self.abl_confidence_update = abl.get("confidence_update", True)
        self.abl_retrieval = abl.get("retrieval", True)
        self.abl_consolidation = abl.get("consolidation", True)
        
        # ---- 运行时计数 ----
        self.total_rounds = 0
        self.reflection_count_by_severity = {"high": 0, "medium": 0, "low": 0}
        self.reflection_hit_count = 0   # 检索到经验的次数
        self.reflection_applied_count = 0   # 实际被推荐器使用的次数
    
    # ============ 反思机制对外的两个接口 ============
    def retrieve_experiences(
        self,
        student_profile,
    ) -> List[ExperienceUnit]:
        """
        给教师 Agent 用: 检索与学生画像相关的历史经验
        """
        if not self.reflection_enabled:
            return []
        
        if not self.abl_retrieval:
            # 消融: 用最近的 N 条经验
            return self.exp_bank.retrieve_recent(top_k=self.retrieval_top_k)
        
        if student_profile.sig_embedding is None:
            return []
        
        results = self.exp_bank.retrieve(
            query_emb=student_profile.sig_embedding,
            top_k=self.retrieval_top_k,
            min_confidence=self.retrieval_min_conf,
            similarity_threshold=self.retrieval_sim_thr,
        )
        if results:
            self.reflection_hit_count += 1
        return results
    
    def process_round(
        self,
        student_profile,
        recommendation: dict,
        simulated_results: dict,
        applied_experiences: List[ExperienceUnit],
        round_id: int = 0,
    ) -> Tuple[Optional[ExperienceUnit], str, dict]:
        """
        一次推荐结束后的反思处理
        Returns:
            (new_exp_or_None, severity, debug_info)
        """
        self.total_rounds += 1
        
        if not self.reflection_enabled:
            return None, "none", {"disabled": True}
        
        # 1) 触发判断
        expected_kg = recommendation.get("expected_kg", 0.0)
        actual_kg = simulated_results.get("actual_kg", 0.0)
        should, severity, debug = self.judge.judge(
            recommendation.get("predicted_correct_rates", []),
            simulated_results.get("simulated_correct", []),
            expected_kg=expected_kg,
            actual_kg=actual_kg,
        )
        
        new_exp = None
        if should:
            # 2) 生成反思
            new_exp = self.engine.reflect(
                student_profile=student_profile,
                recommendation=recommendation,
                simulated_results=simulated_results,
                severity=severity,
                applied_experiences=applied_experiences,
                round_id=round_id,
            )
            if new_exp is not None:
                self.exp_bank.add(new_exp)
                self.reflection_count_by_severity[severity] = \
                    self.reflection_count_by_severity.get(severity, 0) + 1
        
        # 3) 经验置信度回填
        if self.abl_confidence_update:
            for exp in applied_experiences:
                if not should:
                    # 推荐被验证成功 → support
                    self.exp_bank.update_confidence(exp.exp_id, "support")
                elif severity == "high":
                    self.exp_bank.update_confidence(exp.exp_id, "contradict")
                # medium / low 严重度不计入支持/反驳, 仅 use_count 已经在 retrieve 时递增
        
        # 4) 周期性维护
        if self.consolidation_enabled and self.abl_consolidation \
                and self.total_rounds % self.consolidation_interval == 0:
            self.exp_bank.consolidate(
                similarity_threshold=self.consolidation_sim_thr,
                min_cluster_size=self.consolidation_min_cluster,
                tag_overlap_threshold=self.consolidation_tag_thr,
            )
            removed = self.exp_bank.gc()
            logger.info(f"After consolidation+GC: {self.exp_bank.stats()}, removed={removed}")
        
        # 5) 周期性持久化
        if self.total_rounds % self.autosave_interval == 0:
            try:
                self.exp_bank.save(self.persist_path)
            except Exception as e:
                logger.warning(f"Autosave failed: {e}")
        
        return new_exp, severity, debug
    
    # ============ 统计 ============
    def stats(self) -> dict:
        return {
            "total_rounds": self.total_rounds,
            "reflection_enabled": self.reflection_enabled,
            "reflection_by_severity": dict(self.reflection_count_by_severity),
            "reflection_hit_count": self.reflection_hit_count,
            "experience_bank": self.exp_bank.stats(),
            "ablation": {
                "structured_delta": self.abl_structured_delta,
                "confidence_update": self.abl_confidence_update,
                "retrieval": self.abl_retrieval,
                "consolidation": self.abl_consolidation,
            }
        }
    
    def save(self):
        try:
            self.exp_bank.save(self.persist_path)
        except Exception as e:
            logger.warning(f"Save failed: {e}")
