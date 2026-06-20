"""
评估指标实现

两大类指标:
==========
A. 教育推荐核心指标(主要指标)
   - KC-Coverage@K        : 推荐题目覆盖了多少薄弱 KC
   - Expected-KG@K        : 预期学习增益
   - Difficulty-Match@K   : 推荐难度与 ZPD 的匹配度
   - Diversity@K          : 题目多样性

B. 习题级序列推荐参考指标(辅助参考)
   - NDCG / F1 / Hit @K (题目级和 KC 级)
   * 注意: 这类指标对"补弱推荐"场景适配性较弱,主要给审稿人看作对照
"""
from typing import List, Set, Dict, Optional
import math
import numpy as np


# =====================================================================
# A. 主要指标: 教育推荐导向
# =====================================================================

def kc_coverage_at_k(
    recommended_qids: List[str],
    weak_kcs: List[str],
    qid_to_kcs: Dict[str, List[str]],
    K: Optional[int] = None,
) -> float:
    """
    KC Coverage@K
    = |推荐题目覆盖到的薄弱KC集合| / |薄弱KC集合|
    
    衡量"补弱完整度": 推荐够不够覆盖学生薄弱点
    """
    if not weak_kcs:
        return 0.0
    if K is not None:
        recommended_qids = recommended_qids[:K]
    
    weak_set = set(weak_kcs)
    covered = set()
    for qid in recommended_qids:
        for kc in qid_to_kcs.get(qid, []):
            if kc in weak_set:
                covered.add(kc)
    return len(covered) / len(weak_set)


def expected_kg_at_k(
    recommended_qids: List[str],
    weak_kcs: List[str],
    kc_mastery: Dict[str, float],
    qid_to_kcs: Dict[str, List[str]],
    predicted_correct_rates: List[float],
    K: Optional[int] = None,
) -> float:
    """
    Expected Knowledge Gain @ K
    = mean_i ( P(correct_i) * sum_{kc∈C_i ∩ weak} (1 - mastery_kc) )
    
    衡量"潜在补弱效果": 高质量推荐应在薄弱 KC 上有高增益
    """
    if K is not None:
        recommended_qids = recommended_qids[:K]
        predicted_correct_rates = predicted_correct_rates[:K]
    if not recommended_qids:
        return 0.0
    
    weak_set = set(weak_kcs)
    total = 0.0
    for qid, p in zip(recommended_qids, predicted_correct_rates):
        gain = 0.0
        for kc in qid_to_kcs.get(qid, []):
            if kc in weak_set:
                gain += (1.0 - kc_mastery.get(kc, 0.5))
        total += float(p) * gain
    return total / len(recommended_qids)


def difficulty_match_at_k(
    recommended_qids: List[str],
    qid_to_difficulty_rank: Dict[str, int],
    ability_level: str,
    K: Optional[int] = None,
) -> float:
    """
    Difficulty Match @ K
    = 推荐题目中, 难度落在 ZPD 区间(根据能力档动态确定) 的比例
    
    ZPD 区间:
      low    → {1, 2}
      middle → {2, 3}
      high   → {3, 4}
    """
    if K is not None:
        recommended_qids = recommended_qids[:K]
    if not recommended_qids:
        return 0.0
    
    zpd = {
        "low": {1, 2},
        "middle": {2, 3},
        "high": {3, 4},
    }.get(ability_level, {2, 3})
    
    matched = sum(
        1 for qid in recommended_qids
        if qid_to_difficulty_rank.get(qid, 0) in zpd
    )
    return matched / len(recommended_qids)


def diversity_at_k(
    recommended_qids: List[str],
    qid_to_kcs: Dict[str, List[str]],
    K: Optional[int] = None,
) -> float:
    """
    Diversity @ K = unique_KCs / (K * avg_kcs_per_q) 
    
    这里实现的简化版: 不同 KC 数 / K (归一化到 ~[0,1])
    """
    if K is not None:
        recommended_qids = recommended_qids[:K]
    if not recommended_qids:
        return 0.0
    kcs = set()
    for qid in recommended_qids:
        for kc in qid_to_kcs.get(qid, []):
            kcs.add(kc)
    # 一道题平均 ~2 个 KC,理论上限 ≈ 2K
    return min(1.0, len(kcs) / (2 * len(recommended_qids)))


# =====================================================================
# B. 习题级序列推荐参考指标
# =====================================================================

def _dcg(rels: List[float]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def ndcg_at_k(
    recommended_qids: List[str],
    ground_truth_qids: List[str],
    K: int,
) -> float:
    """
    NDCG@K (题目级)
    GT 为正确答对的题集合(理解为"应该推荐"的)
    """
    gt_set = set(ground_truth_qids)
    if not gt_set:
        return 0.0
    rels = [1.0 if qid in gt_set else 0.0 for qid in recommended_qids[:K]]
    dcg = _dcg(rels)
    n = min(K, len(gt_set))
    idcg = _dcg([1.0] * n)
    return dcg / idcg if idcg > 0 else 0.0


def f1_at_k(
    recommended_qids: List[str],
    ground_truth_qids: List[str],
    K: int,
) -> float:
    rec_set = set(recommended_qids[:K])
    gt_set = set(ground_truth_qids)
    if not rec_set or not gt_set:
        return 0.0
    tp = len(rec_set & gt_set)
    if tp == 0:
        return 0.0
    prec = tp / len(rec_set)
    rec = tp / len(gt_set)
    return 2 * prec * rec / (prec + rec)


def hit_at_k(
    recommended_qids: List[str],
    ground_truth_qids: List[str],
    K: int,
) -> float:
    rec_set = set(recommended_qids[:K])
    gt_set = set(ground_truth_qids)
    return 1.0 if rec_set & gt_set else 0.0


# 对应的 KC 级别版本

def ndcg_at_k_kc(
    recommended_qids: List[str],
    ground_truth_kcs: List[str],
    qid_to_kcs: Dict[str, List[str]],
    K: int,
) -> float:
    gt_set = set(ground_truth_kcs)
    if not gt_set:
        return 0.0
    rels = []
    for qid in recommended_qids[:K]:
        kcs = set(qid_to_kcs.get(qid, []))
        rels.append(1.0 if (kcs & gt_set) else 0.0)
    dcg = _dcg(rels)
    idcg = _dcg([1.0] * min(K, len(gt_set)))
    return dcg / idcg if idcg > 0 else 0.0


def f1_at_k_kc(
    recommended_qids: List[str],
    ground_truth_kcs: List[str],
    qid_to_kcs: Dict[str, List[str]],
    K: int,
) -> float:
    gt_set = set(ground_truth_kcs)
    rec_kc_set = set()
    for qid in recommended_qids[:K]:
        for kc in qid_to_kcs.get(qid, []):
            rec_kc_set.add(kc)
    if not rec_kc_set or not gt_set:
        return 0.0
    tp = len(rec_kc_set & gt_set)
    if tp == 0:
        return 0.0
    prec = tp / len(rec_kc_set)
    rec = tp / len(gt_set)
    return 2 * prec * rec / (prec + rec)


def hit_at_k_kc(
    recommended_qids: List[str],
    ground_truth_kcs: List[str],
    qid_to_kcs: Dict[str, List[str]],
    K: int,
) -> float:
    gt_set = set(ground_truth_kcs)
    for qid in recommended_qids[:K]:
        kcs = set(qid_to_kcs.get(qid, []))
        if kcs & gt_set:
            return 1.0
    return 0.0
