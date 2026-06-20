"""测试 evaluation 指标"""
import pytest
from evaluation.metrics import (
    kc_coverage_at_k, expected_kg_at_k, difficulty_match_at_k, diversity_at_k,
    ndcg_at_k, f1_at_k, hit_at_k,
)


def test_kc_coverage_full():
    qid_to_kcs = {"q1": ["A"], "q2": ["B"], "q3": ["C"]}
    cov = kc_coverage_at_k(["q1", "q2", "q3"], ["A", "B"], qid_to_kcs, K=3)
    assert cov == 1.0


def test_kc_coverage_partial():
    qid_to_kcs = {"q1": ["A"], "q2": ["X"]}
    cov = kc_coverage_at_k(["q1", "q2"], ["A", "B"], qid_to_kcs, K=2)
    assert cov == 0.5


def test_kc_coverage_empty_weak():
    cov = kc_coverage_at_k(["q1"], [], {}, K=1)
    assert cov == 0.0


def test_expected_kg():
    qid_to_kcs = {"q1": ["A"], "q2": ["B"]}
    mastery = {"A": 0.2, "B": 0.3}
    weak = ["A", "B"]
    # 题 q1: P=0.5, gain=0.8 → contribution=0.4
    # 题 q2: P=0.5, gain=0.7 → contribution=0.35
    # mean = 0.375
    val = expected_kg_at_k(["q1", "q2"], weak, mastery, qid_to_kcs, [0.5, 0.5], K=2)
    assert abs(val - 0.375) < 1e-6


def test_difficulty_match_middle():
    qid_to_diff = {"q1": 2, "q2": 3, "q3": 5}
    # middle 的 ZPD = {2,3}, 命中 2 个,共 3 个题
    m = difficulty_match_at_k(["q1", "q2", "q3"], qid_to_diff, "middle", K=3)
    assert abs(m - 2/3) < 1e-6


def test_diversity():
    qid_to_kcs = {"q1": ["A"], "q2": ["A"], "q3": ["A"]}
    d_low = diversity_at_k(["q1", "q2", "q3"], qid_to_kcs, K=3)
    
    qid_to_kcs2 = {"q1": ["A", "B"], "q2": ["C"], "q3": ["D"]}
    d_high = diversity_at_k(["q1", "q2", "q3"], qid_to_kcs2, K=3)
    
    assert d_high > d_low


def test_ndcg():
    rec = ["q1", "q2", "q3"]
    gt = ["q2", "q5"]
    val = ndcg_at_k(rec, gt, K=3)
    assert 0 <= val <= 1


def test_f1():
    val = f1_at_k(["q1", "q2", "q3"], ["q1", "q4"], K=3)
    # 命中1, prec=1/3, rec=1/2, f1=2*1/3*1/2/(1/3+1/2)
    expected = 2 * (1/3) * (1/2) / (1/3 + 1/2)
    assert abs(val - expected) < 1e-6


def test_hit():
    assert hit_at_k(["q1", "q2"], ["q3"], K=2) == 0.0
    assert hit_at_k(["q1", "q2"], ["q2"], K=2) == 1.0
