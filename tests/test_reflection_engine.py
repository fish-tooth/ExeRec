"""测试 ReflectionEngine"""
import pytest
import numpy as np
from reflection import ReflectionEngine, TriggerJudge


def test_trigger_judge_no_trigger():
    judge = TriggerJudge()
    # 预测和实际接近: |0.9-1|+|0.8-1|+|0.2-0| 的均值约 0.1 → 不应触发 high
    should, sev, debug = judge.judge(
        predicted_probs=[0.9, 0.8, 0.2],
        simulated_correct=[1, 1, 0],
        expected_kg=0.05, actual_kg=0.04,
    )
    assert sev in ("none", "low")


def test_trigger_judge_high():
    judge = TriggerJudge()
    # 大幅误差应触发 high
    should, sev, debug = judge.judge(
        predicted_probs=[0.9, 0.9, 0.9],
        simulated_correct=[0, 0, 0],
        expected_kg=0.20, actual_kg=0.0,
    )
    assert should is True
    assert sev == "high"


def test_trigger_judge_systematic_bias():
    """全错且预测乐观 → high"""
    judge = TriggerJudge()
    should, sev, debug = judge.judge(
        predicted_probs=[0.8, 0.7, 0.75],
        simulated_correct=[0, 0, 0],
        expected_kg=0.10, actual_kg=0.0,
    )
    assert should is True
    assert sev == "high"


def test_reflection_parse_valid():
    """测试 JSON 解析"""
    content = """
    {
      "lesson": "推荐过难,需降低难度",
      "tags": ["overestimate_difficulty"],
      "action_delta": {
        "difficulty_shift": -1,
        "prereq_check": true,
        "diversity_target": 0.5,
        "kc_focus_change": []
      },
      "self_confidence": 0.7,
      "contradicts_prior": false
    }
    """
    parsed = ReflectionEngine._parse_and_validate(content)
    assert parsed is not None
    assert parsed["lesson"]
    assert parsed["action_delta"]["difficulty_shift"] == -1
    assert parsed["self_confidence"] == 0.7


def test_reflection_parse_invalid_tag():
    """非法 tag 应过滤"""
    content = """
    {
      "lesson": "ok",
      "tags": ["invalid_tag", "overestimate_difficulty"],
      "action_delta": {"difficulty_shift": 0, "prereq_check": false, "diversity_target": 0.4, "kc_focus_change": []},
      "self_confidence": 0.5,
      "contradicts_prior": false
    }
    """
    parsed = ReflectionEngine._parse_and_validate(content)
    assert "invalid_tag" not in parsed["tags"]
    assert "overestimate_difficulty" in parsed["tags"]


def test_reflection_parse_clip():
    """超界值应被截断"""
    content = """
    {
      "lesson": "ok",
      "tags": ["other"],
      "action_delta": {
        "difficulty_shift": 100,
        "prereq_check": true,
        "diversity_target": 2.0,
        "kc_focus_change": []
      },
      "self_confidence": 5.0,
      "contradicts_prior": false
    }
    """
    parsed = ReflectionEngine._parse_and_validate(content)
    assert -2 <= parsed["action_delta"]["difficulty_shift"] <= 2
    assert 0 <= parsed["action_delta"]["diversity_target"] <= 1
    assert 0 <= parsed["self_confidence"] <= 1


def test_reflection_parse_malformed():
    """坏 JSON 应返回 None,不抛异常"""
    parsed = ReflectionEngine._parse_and_validate("this is not json")
    assert parsed is None


def test_reflection_full_flow(mock_llm, fake_student_profile):
    """从 mock LLM 输出生成完整 ExperienceUnit"""
    engine = ReflectionEngine(mock_llm)
    recommendation = {
        "questions": ["q1", "q2", "q3"],
        "predicted_correct_rates": [0.7, 0.6, 0.5],
        "kc_focus": ["KC_A"],
        "difficulty_distribution": {"中等": 3},
        "strategy_label": "weak_kc_first",
        "rationale": "test rationale",
        "expected_kg": 0.10,
    }
    sim = {
        "simulated_correct": [0, 0, 0],
        "actual_kg": 0.0,
    }
    exp = engine.reflect(
        student_profile=fake_student_profile,
        recommendation=recommendation,
        simulated_results=sim,
        severity="high",
        applied_experiences=[],
        round_id=1,
    )
    assert exp is not None
    assert exp.lesson
    assert 0 <= exp.confidence <= 1
    assert exp.severity == "high"
