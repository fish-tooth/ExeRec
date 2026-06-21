"""
Evaluator
- 按 K 列表批量计算所有指标
- 聚合多学生结果为总报告
- 同时追踪反思机制相关的系统级观测量
"""
import json
from pathlib import Path
import time
from typing import List, Dict, Optional
from collections import defaultdict
import numpy as np

from .metrics import (
    kc_coverage_at_k, expected_kg_at_k, difficulty_match_at_k, diversity_at_k,
    ndcg_at_k, f1_at_k, hit_at_k, ndcg_at_k_kc, f1_at_k_kc, hit_at_k_kc,
)
from utils.logger import get_logger

logger = get_logger("eval.evaluator")


class Evaluator:
    """
    评估单元
    
    使用方式:
        evaluator = Evaluator(cfg, question_bank)
        for student_id in students:
            evaluator.add_round(student_id, profile, recommendation, ground_truth)
        report = evaluator.summarize()
        evaluator.save(report)
    """
    
    def __init__(
        self,
        cfg: dict,
        question_bank,
    ):
        self.cfg = cfg
        self.qb = question_bank
        
        eval_cfg = cfg.get("evaluation", {})
        self.K_list = eval_cfg.get("K_list", [1, 3, 5, 10, 20])
        self.metric_flags = eval_cfg.get("metrics", {})
        self.output_dir = eval_cfg.get("output_dir", "./logs/eval_results")
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # 学生级累计
        self.rounds = []   # 每条 round 的指标
        # 反思相关追踪
        self.reflection_apply_count = 0   # 推荐时使用了经验的次数
        self.reflection_effect_count = 0  # 使用经验后表现优于不使用的次数(需双轨实验)
    
    def add_round(
        self,
        student_id: str,
        student_profile,
        recommendation: dict,
        ground_truth: Optional[dict] = None,
    ):
        """
        添加一轮推荐及其评估结果
        
        Args:
            ground_truth: 可选,包含
                - "qids": 学生未来答对的题(用于 NDCG/F1/Hit)
                - "kcs": 学生未来涉及的 KC
                若为 None,只计算不依赖 GT 的核心指标
        """
        rec_qids = recommendation.get("questions", [])
        probs = recommendation.get("predicted_correct_rates", [])
        
        # 构建 qid → KC 与 qid → 难度等级 的映射(局部 cache)
        qid_to_kcs = {qid: self.qb.get_kcs_of(qid) for qid in rec_qids}
        qid_to_diff = {qid: self.qb.difficulty_rank(qid) for qid in rec_qids}
        
        round_metrics = {
            "student_id": student_id,
            "n_recommended": len(rec_qids),
        }
        
        weak_kcs = student_profile.weak_kcs
        mastery = student_profile.kc_mastery
        ability = student_profile.ability_level
        
        for K in self.K_list:
            # ---- 主要指标 ----
            if self.metric_flags.get("kc_coverage", True):
                round_metrics[f"kc_coverage@{K}"] = kc_coverage_at_k(
                    rec_qids, weak_kcs, qid_to_kcs, K
                )
            if self.metric_flags.get("expected_kg", True):
                round_metrics[f"expected_kg@{K}"] = expected_kg_at_k(
                    rec_qids, weak_kcs, mastery, qid_to_kcs, probs, K
                )
            if self.metric_flags.get("difficulty_match", True):
                round_metrics[f"difficulty_match@{K}"] = difficulty_match_at_k(
                    rec_qids, qid_to_diff, ability, K
                )
            if self.metric_flags.get("diversity", True):
                round_metrics[f"diversity@{K}"] = diversity_at_k(rec_qids, qid_to_kcs, K)
            
            # ---- 参考指标 (需要 GT) ----
            if ground_truth:
                gt_qids = ground_truth.get("qids", [])
                gt_kcs = ground_truth.get("kcs", [])
                if self.metric_flags.get("ndcg_q", True):
                    round_metrics[f"ndcg_q@{K}"] = ndcg_at_k(rec_qids, gt_qids, K)
                if self.metric_flags.get("f1_q", True):
                    round_metrics[f"f1_q@{K}"] = f1_at_k(rec_qids, gt_qids, K)
                if self.metric_flags.get("hit_q", True):
                    round_metrics[f"hit_q@{K}"] = hit_at_k(rec_qids, gt_qids, K)
                if self.metric_flags.get("ndcg_kc", True):
                    round_metrics[f"ndcg_kc@{K}"] = ndcg_at_k_kc(rec_qids, gt_kcs, qid_to_kcs, K)
                if self.metric_flags.get("f1_kc", True):
                    round_metrics[f"f1_kc@{K}"] = f1_at_k_kc(rec_qids, gt_kcs, qid_to_kcs, K)
                if self.metric_flags.get("hit_kc", True):
                    round_metrics[f"hit_kc@{K}"] = hit_at_k_kc(rec_qids, gt_kcs, qid_to_kcs, K)
        
        # ---- 反思应用追踪 ----
        applied_ids = recommendation.get("applied_experience_ids", [])
        round_metrics["used_experience_count"] = len(applied_ids)
        if applied_ids:
            self.reflection_apply_count += 1
        
        self.rounds.append(round_metrics)
    
    def summarize(self, global_agent_stats: Optional[dict] = None) -> dict:
        """聚合所有 round → 单数字指标"""
        if not self.rounds:
            return {"n_students": 0}
        
        # 跨学生平均
        keys = set()
        for r in self.rounds:
            keys.update(r.keys())
        keys -= {"student_id"}
        
        agg = {"n_students": len(self.rounds)}
        for k in sorted(keys):
            vals = [r.get(k) for r in self.rounds if isinstance(r.get(k), (int, float))]
            if not vals:
                continue
            agg[f"{k}_mean"] = float(np.mean(vals))
            agg[f"{k}_std"] = float(np.std(vals))
        
        # 反思应用率
        if len(self.rounds) > 0:
            agg["reflection_apply_rate"] = (
                self.reflection_apply_count / len(self.rounds)
            )
        
        # 接入 GlobalAgent 的运行时状态
        if global_agent_stats:
            agg["global_agent"] = global_agent_stats
        
        return agg
    
    def save(self, summary: dict, name: str = "eval_summary"):

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        """保存详细 rounds 和聚合结果"""
        out_dir = Path(self.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        with open(out_dir / f"{name}_rounds_{timestamp}.json", "w", encoding="utf-8") as f:
            json.dump(self.rounds, f, ensure_ascii=False, indent=2, default=str)
        
        with open(out_dir / f"{name}_summary_{timestamp}.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"Evaluation saved to {out_dir}")
