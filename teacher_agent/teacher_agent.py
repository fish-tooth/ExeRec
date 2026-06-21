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
        # MMR 多样性重排: λ ∈ [0,1], 越大越偏相关性,越小越偏多样性
        # 0.7 是相关性与多样性的典型平衡点
        self.mmr_lambda = ta_cfg.get("mmr_lambda", 0.7)
        
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
        
        # ★ 候选不足 K*4 时,逐步放宽:
        #   1. 扩展薄弱 KC 范围(top_n → top_n+5)
        #   2. 放宽难度限制(全难度)
        #   3. 用学生未涉及但与薄弱 KC 相邻的 KC
        if len(l1_pool) < K * 4:
            # 扩展薄弱 KC: 从画像取所有 KC,而非只用 strategy.kc_focus
            extended_kcs = list(student_profile.weak_kcs)
            if not extended_kcs:
                # 取掌握度最低的 10 个
                sorted_kcs = sorted(
                    student_profile.kc_mastery.items(),
                    key=lambda x: x[1]
                )
                extended_kcs = [kc for kc, _ in sorted_kcs[:10]]
            
            l1_extended = self.pipeline.level1(
                weak_kcs=extended_kcs[:10],
                difficulty_centers=diff_centers,
                exclude_qids=exclude,
            )
            for qid in l1_extended:
                if qid not in l1_pool:
                    l1_pool.append(qid)
        
        # 仍然不足: 放宽难度
        if len(l1_pool) < K * 2:
            l1_relaxed = self.qb.filter(
                kcs=student_profile.weak_kcs[:10] if student_profile.weak_kcs else None,
                exclude_qids=exclude,
            )
            for qid in l1_relaxed:
                if qid not in l1_pool:
                    l1_pool.append(qid)
                    if len(l1_pool) >= K * 6:
                        break
        
        # 5) L2 向量精排
        student_emb = student_profile.fused_embedding
        if student_emb is None:
            student_emb = student_profile.individual_embedding
        if student_emb is None:
            student_emb = np.zeros(1024, dtype=np.float32)
        
        weak_kc_emb = self.es.get_kcs_emb(student_profile.weak_kcs)
        
        # ZPD 难度集合(根据能力档)
        zpd_set = {
            "low": {1, 2},
            "middle": {2, 3},
            "high": {3, 4},
        }.get(student_profile.ability_level, {2, 3})
        
        l2_ranked = self.pipeline.level2(
            candidates=l1_pool,
            student_emb=student_emb,
            weak_kc_emb=weak_kc_emb,
            zpd_difficulty_set=zpd_set,
            weak_kcs_set=set(student_profile.weak_kcs),
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
            k_target=K,
        )
        
        selected_qids = l3_result["selected_qids"][:K]
        probs = l3_result["predicted_correct_rates"][:K]
        # 长度对齐
        while len(probs) < len(selected_qids):
            probs.append(0.5)
        
        # 6.5) MMR 重排: 在保持相关性的同时,避免连续推荐同 KC 的题目
        # 这显著提升 NDCG@K 在大 K 下的稳定性,也提升 KC 多样性
        selected_qids, probs = self._mmr_rerank(
            selected_qids, probs,
            weak_kcs=student_profile.weak_kcs,
            kc_mastery=student_profile.kc_mastery,
            lambda_=self.mmr_lambda,
        )
        
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
    
    # ============ MMR 多样性重排 ============
    def _mmr_rerank(
        self,
        qids: List[str],
        probs: List[float],
        weak_kcs: List[str],
        kc_mastery: Dict[str, float],
        lambda_: float = 0.7,
    ) -> tuple:
        """
        MMR 重排,降低 KC 重复,保持薄弱 KC 覆盖优先
        
        每一步选择 argmax_q [ λ * relevance(q) - (1-λ) * redundancy(q | selected) ]
        其中:
          - relevance(q) = predicted_correct_prob * weak_kc_uncovered_gain
          - redundancy(q | selected) = q 与已选题目的 KC 重合度的最大值
        
        效果:
          - 若已选了"求回归方程"的题,下一道选其他薄弱 KC 的题获得 bonus
          - 避免连续 3 道全是同一 KC,提升 NDCG@K 在大 K 下的表现
          - 在保持相关性的前提下提升 diversity
        """
        if not qids or len(qids) <= 1:
            return qids, probs
        
        weak_set = set(weak_kcs)
        
        # 预先算每道题的 KC 集合和"相关性得分"
        qid_to_kc_set = {qid: set(self.qb.get_kcs_of(qid)) for qid in qids}
        
        def relevance(qid: str, p: float) -> float:
            """题目相关性 = 学生作对概率 * 该题覆盖的薄弱 KC 的剩余学习空间"""
            kcs = qid_to_kc_set[qid] & weak_set
            if not kcs:
                # 不覆盖薄弱 KC 的题,相关性大幅惩罚
                return 0.1 * p
            # 该题每个薄弱 KC 的"还需学习度"
            gain = sum(1.0 - kc_mastery.get(kc, 0.5) for kc in kcs)
            return p * gain
        
        # 第一步:选相关性最高的
        scores = [(qid, p, relevance(qid, p)) for qid, p in zip(qids, probs)]
        scores.sort(key=lambda x: x[2], reverse=True)
        
        # 维护已选集合的 KC 累积覆盖度(为了重复惩罚)
        selected = [scores[0][0]]
        selected_probs = [scores[0][1]]
        selected_kc_count: Dict[str, int] = {}
        for kc in qid_to_kc_set[selected[0]]:
            selected_kc_count[kc] = selected_kc_count.get(kc, 0) + 1
        
        remaining = [(qid, p, rel) for qid, p, rel in scores[1:]]
        
        while remaining and len(selected) < len(qids):
            best_mmr = -float("inf")
            best_idx = 0
            for i, (qid, p, rel) in enumerate(remaining):
                # 冗余度: 该题的 KC 在已选集合中出现次数的最大值
                # (出现越多次,越冗余)
                kcs = qid_to_kc_set[qid]
                if not kcs:
                    redundancy = 0.0
                else:
                    max_overlap = max(
                        (selected_kc_count.get(kc, 0) for kc in kcs),
                        default=0
                    )
                    redundancy = float(max_overlap)
                
                mmr = lambda_ * rel - (1 - lambda_) * redundancy
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = i
            
            chosen_qid, chosen_p, _ = remaining.pop(best_idx)
            selected.append(chosen_qid)
            selected_probs.append(chosen_p)
            for kc in qid_to_kc_set[chosen_qid]:
                selected_kc_count[kc] = selected_kc_count.get(kc, 0) + 1
        
        return selected, selected_probs
    
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
