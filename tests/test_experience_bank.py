"""测试 ExperienceBank: CRUD/检索/置信度更新/持久化"""
import pytest
import numpy as np
from reflection import ExperienceBank, ExperienceUnit


@pytest.fixture
def empty_bank():
    return ExperienceBank(retire_below=0.20, dormant_below=0.35)


@pytest.fixture
def populated_bank():
    bank = ExperienceBank(retire_below=0.20, dormant_below=0.35)
    rng = np.random.RandomState(0)
    for i in range(10):
        emb = rng.randn(32).astype(np.float32)
        # 后5条与前5条相似(为了测合并)
        if i >= 5:
            emb = rng.randn(32).astype(np.float32) * 0.1 + np.ones(32) * 0.5
        exp = ExperienceUnit(
            student_signature={
                "weak_kcs": [f"KC_{i%3}"],
                "ability_level": "middle",
                "learning_preference": "step_by_step",
            },
            student_sig_embedding=emb,
            lesson=f"Lesson {i}",
            lesson_tags=["overestimate_difficulty"],
            suggested_action_delta={
                "difficulty_shift": -1, "prereq_check": True,
                "diversity_target": 0.4, "kc_focus_change": [],
            },
            confidence=0.5 + (i % 3) * 0.1,
        )
        bank.add(exp)
    return bank


def test_add_and_get(empty_bank):
    exp = ExperienceUnit(lesson="test")
    empty_bank.add(exp)
    assert empty_bank.size() == 1
    assert empty_bank.get(exp.exp_id) is exp


def test_retrieve(populated_bank):
    # 查询向量接近后 5 条
    query = np.ones(32, dtype=np.float32) * 0.5
    results = populated_bank.retrieve(
        query_emb=query, top_k=3,
        min_confidence=0.0, similarity_threshold=0.0,
    )
    assert len(results) <= 3
    # 应该返回的是相似度较高的
    assert all(isinstance(r, ExperienceUnit) for r in results)


def test_retrieve_recent(populated_bank):
    """消融模式: retrieve_recent 不基于向量"""
    results = populated_bank.retrieve_recent(top_k=5)
    assert len(results) == 5
    # 应该按 created_at 倒序
    times = [r.created_at for r in results]
    assert times == sorted(times, reverse=True)


def test_confidence_update_support(empty_bank):
    exp = ExperienceUnit(confidence=0.5, support_count=0, contradict_count=0)
    empty_bank.add(exp)
    for _ in range(5):
        empty_bank.update_confidence(exp.exp_id, "support")
    refreshed = empty_bank.get(exp.exp_id)
    assert refreshed.support_count == 5
    # 多次 support 应该提升置信度
    assert refreshed.confidence > 0.5
    assert refreshed.status == "active"


def test_confidence_update_contradict_retires(empty_bank):
    """反复 contradict 应让经验进入 retired"""
    exp = ExperienceUnit(confidence=0.5, support_count=0, contradict_count=0)
    empty_bank.add(exp)
    for _ in range(20):
        empty_bank.update_confidence(exp.exp_id, "contradict")
    refreshed = empty_bank.get(exp.exp_id)
    assert refreshed.confidence < 0.35
    assert refreshed.status in ("dormant", "retired")


def test_save_and_load(populated_bank, tmp_path):
    path = tmp_path / "bank.pkl"
    populated_bank.save(str(path))
    
    new_bank = ExperienceBank()
    new_bank.load(str(path))
    assert new_bank.size() == populated_bank.size()


def test_gc(populated_bank):
    # 强制把其中一些设为 retired
    exps = list(populated_bank.experiences.values())
    exps[0].status = "retired"
    exps[1].status = "retired"
    removed = populated_bank.gc()
    assert removed == 2


def test_stats(populated_bank):
    s = populated_bank.stats()
    assert s["total"] == 10
    assert s["active"] >= 0
    assert "avg_confidence" in s
