"""
教师 Agent 主类
推荐流程:
  1) 检索相关反思经验
  2) 聚合反思 → 策略修正
  3) 三级检索流水线
  4) 计算预期学习增益
  5) 返回推荐结果
"""
import numpy as np
from collections import Counter
from typing import List, Dict, Optional

from .retrieval_pipeline import RetrievalPipeline, DIFFICULTY_NAMES
from .strategy_aggregator import StrategyAggregator
from data_processing import QuestionBank, EmbeddingStore
from llm_adapter import LLMFactory
from reflection import ExperienceUnit
from utils.logger import get_logger

logger = get_logger("teacher.agent")


class TeacherAgent:
    """
    教师 Agent
    
    核心入口: recommend(student_profile, relevant_experiences, K)
    """
    
    def __init__(
        self,
        cfg: dict,
        question_bank: QuestionBank,
        embedding_store: EmbeddingStore,
        llm: LLMFactory,
    ):
        self.cfg = cfg
        self.qb = question_bank
        self.es = embedding_store
        self.llm = llm
        
        ta_cfg = cfg.get("teacher_agent", {})
        ret_cfg = ta_cfg.get("retrieval", {})
        zpd_cfg = ta_cfg.get("zpd", {})
        
        self.pipeline = RetrievalPipeline(
            question_bank=question_bank,
            embedding_store=embedding_store,
            llm=llm,
            level1_pool_size=ret_cfg.get("level1_pool_size", 200),
            level2_top_n=ret_cfg.get("level2_top_n", 50),
            level3_top_k=ret_cfg.get("level3_top_k", 5),
        )
        
        self.zpd_lower = zpd_cfg.get("lower", 0.5)
        self.zpd_upper = zpd_cfg.get("upper", 0.75)
        self.default_diversity_target = ta_cfg.get("diversity_target", 0.4)
        
        # 消融开关从 reflection.ablation 读
        abl = cfg.get("reflection", {}).get("ablation", {})
        self.use_structured_delta = abl.get("structured_delta", True)
    
    # ============ 主接口 ============
    def recommend(
        self,
        student_profile,
        relevant_experiences: Optional[List[ExperienceUnit]] = None,
        K: int = 5,
    ) -> Dict:
        """
        给 student_profile 生成 K 个推荐
        
        Returns:
            {
              "questions": [qid1, ..., qidK],
              "predicted_correct_rates": [...],
              "kc_focus": [...],
              "difficulty_distribution": {"容易":2, ...},
              "diversity_score": float,
              "strategy_label": str,
              "rationale": str,
              "applied_experience_ids": [...],
              "expected_kg": float,
            }
        """
        relevant_experiences = relevant_experiences or []
        
        # 1) 基础策略
        base_strategy = self._base_strategy(student_profile)
        
        # 2) 反思聚合
        delta = StrategyAggregator.aggregate(
            relevant_experiences,
            use_structured_delta=self.use_structured_delta,
        )
        
        # 3) 应用 delta
        strategy = self._apply_delta(base_strategy, delta)
        
        # 4) L1 召回
        exclude = set(student_profile.answered_qids)
        diff_centers = self._strategy_to_diff_centers(strategy)
        l1_pool = self.pipeline.level1(
            weak_kcs=strategy["kc_focus"],
            difficulty_centers=diff_centers,
            exclude_qids=exclude,
        )
        
        # 5) L2 向量精排
        student_emb = student_profile.fused_embedding
        if student_emb is None:
            student_emb = student_profile.individual_embedding
        if student_emb is None:
            student_emb = np.zeros(1024, dtype=np.float32)
        
        weak_kc_emb = self.es.get_kcs_emb(student_profile.weak_kcs)
        
        l2_ranked = self.pipeline.level2(
            candidates=l1_pool,
            student_emb=student_emb,
            weak_kc_emb=weak_kc_emb,
        )
        
        # 候选不足时, 用 L1 顺序补足
        if len(l2_ranked) < K:
            covered = {q for q, _ in l2_ranked}
            for qid in l1_pool:
                if qid not in covered:
                    l2_ranked.append((qid, 0.0))
                    if len(l2_ranked) >= K * 3:
                        break
        
        if not l2_ranked:
            logger.warning(f"Empty candidates for student {student_profile.student_id}; fallback to random")
            return self._fallback_recommendation(student_profile, K, exclude)
        
        # 6) L3 LLM 重排序 + 理由
        lessons_text = StrategyAggregator.lessons_text(
            relevant_experiences, min_confidence=0.5
        )
        l3_result = self.pipeline.level3(
            student_profile=student_profile,
            candidates_with_score=l2_ranked,
            strategy=strategy,
            prior_lessons_text=lessons_text,
        )
        
        selected_qids = l3_result["selected_qids"][:K]
        probs = l3_result["predicted_correct_rates"][:K]
        # 长度对齐
        while len(probs) < len(selected_qids):
            probs.append(0.5)
        
        # 7) 统计输出
        diff_dist = self._difficulty_distribution(selected_qids)
        diversity = self._compute_diversity(selected_qids)
        expected_kg = self._expected_kg(student_profile, selected_qids, probs)
        
        return {
            "questions": selected_qids,
            "predicted_correct_rates": probs,
            "kc_focus": strategy["kc_focus"],
            "difficulty_distribution": diff_dist,
            "diversity_score": diversity,
            "strategy_label": l3_result.get("strategy_label", "default"),
            "rationale": l3_result.get("rationale", ""),
            "applied_experience_ids": delta.get("used_exp_ids", []),
            "expected_kg": expected_kg,
            "_strategy_internal": strategy,
        }
    
    # ============ 基础策略 ============
    def _base_strategy(self, profile) -> dict:
        """根据画像生成基础策略"""
        # 默认聚焦薄弱 KC
        kc_focus = list(profile.weak_kcs[:3]) if profile.weak_kcs else []
        
        # ZPD 难度: 根据能力档反推
        if profile.ability_level == "low":
            diff_target = 2  # 中等
        elif profile.ability_level == "high":
            diff_target = 3  # 较难
        else:
            diff_target = 2
        
        return {
            "kc_focus": kc_focus,
            "difficulty_target": diff_target,
            "diversity_target": self.default_diversity_target,
            "prereq_check": False,
        }
    
    def _apply_delta(self, base: dict, delta: dict) -> dict:
        result = dict(base)
        # difficulty_shift
        ds = delta.get("difficulty_shift", 0)
        result["difficulty_target"] = max(1, min(5, base["difficulty_target"] + ds))
        # prereq_check
        result["prereq_check"] = bool(delta.get("prereq_check", False) or base.get("prereq_check", False))
        # diversity_target
        if delta.get("diversity_target") is not None:
            result["diversity_target"] = delta["diversity_target"]
        # kc_focus_change(扩充而非替换)
        for kc in (delta.get("kc_focus_change") or []):
            if kc not in result["kc_focus"]:
                result["kc_focus"].append(kc)
        return result
    
    def _strategy_to_diff_centers(self, strategy: dict) -> List[int]:
        """难度目标 → 一组邻近难度档(便于检索容错)"""
        center = strategy.get("difficulty_target", 2)
        if center == 1:
            return [1, 2]
        if center == 5:
            return [4, 5]
        return [center - 1, center, center + 1]
    
    # ============ 输出统计 ============
    def _difficulty_distribution(self, qids: List[str]) -> Dict[str, int]:
        counter = Counter()
        for qid in qids:
            q = self.qb.get(qid)
            if q:
                counter[q.get("difficulty", "未知")] += 1
        return dict(counter)
    
    def _compute_diversity(self, qids: List[str]) -> float:
        """KC 多样性 = 涉及的 KC 数 / 题目数"""
        if not qids:
            return 0.0
        kcs = set()
        for qid in qids:
            for kc in self.qb.get_kcs_of(qid):
                kcs.add(kc)
        return len(kcs) / max(1, len(qids)) / 3.0  # 归一化到 ~[0,1]
    
    def _expected_kg(
        self,
        profile,
        qids: List[str],
        probs: List[float],
    ) -> float:
        """
        预期学习增益:
        对每道题 i, 涉及 KC 集合 C_i,
        增益 g_i = P(correct_i) * sum_{kc in C_i & weak_kcs} (1 - mastery[kc])
        总增益 = mean(g_i)
        """
        weak_set = set(profile.weak_kcs)
        mastery = profile.kc_mastery or {}
        total = 0.0
        n = 0
        for qid, p in zip(qids, probs):
            kcs = self.qb.get_kcs_of(qid)
            gain = 0.0
            for kc in kcs:
                if kc in weak_set:
                    gain += (1.0 - mastery.get(kc, 0.5))
            total += float(p) * gain
            n += 1
        return total / max(1, n)
    
    def _fallback_recommendation(self, profile, K, exclude) -> Dict:
        """空候选时的兜底"""
        all_qids = [q for q in self.qb.all_qids if q not in exclude]
        import random
        random.seed(42)
        picked = random.sample(all_qids, min(K, len(all_qids)))
        return {
            "questions": picked,
            "predicted_correct_rates": [0.5] * len(picked),
            "kc_focus": profile.weak_kcs[:3],
            "difficulty_distribution": self._difficulty_distribution(picked),
            "diversity_score": self._compute_diversity(picked),
            "strategy_label": "fallback_random",
            "rationale": "候选为空,采用随机兜底",
            "applied_experience_ids": [],
            "expected_kg": 0.0,
            "_strategy_internal": self._base_strategy(profile),
        }
