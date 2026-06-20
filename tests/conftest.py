"""pytest 公共 fixture"""
import sys
from pathlib import Path

# 把项目根加入 sys.path,使所有测试能 import
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import numpy as np
from llm_adapter import LLMFactory


@pytest.fixture
def mock_cfg():
    """最小可用配置"""
    return {
        "experiment": {"seed": 42, "name": "test"},
        "llm": {
            "default_backend": "mock",
            "backends": {"mock": {}},
            "cache": {"enable": False, "cache_dir": "/tmp/test_cache"},
            "temperature": 0.3,
            "max_tokens": 512,
        },
        "reflection": {
            "enable": True,
            "triggers": {
                "mae_high": 0.40, "mae_medium": 0.25, "mae_low": 0.15,
                "kg_gap_high": 0.15, "kg_gap_medium": 0.08,
            },
            "retrieval": {"top_k": 3, "min_confidence": 0.4, "similarity_threshold": 0.6},
            "confidence": {"decay_per_day": 0.99, "retire_below": 0.20, "dormant_below": 0.35},
            "consolidation": {
                "enable": True, "interval": 100,
                "similarity_threshold": 0.9, "min_cluster_size": 3,
                "tag_overlap_threshold": 0.6,
            },
            "persist": {"path": "/tmp/test_exp_bank.pkl", "autosave_interval": 50},
            "ablation": {
                "structured_delta": True, "confidence_update": True,
                "retrieval": True, "consolidation": True,
            },
        },
    }


@pytest.fixture
def mock_llm(mock_cfg):
    return LLMFactory.from_config(mock_cfg)


@pytest.fixture
def fake_student_profile():
    """构造一个最小的 StudentProfile"""
    from student_agent import StudentProfile
    p = StudentProfile(
        student_id="test_student_001",
        kc_mastery={"KC_A": 0.2, "KC_B": 0.3, "KC_C": 0.9},
        weak_kcs=["KC_A", "KC_B"],
        ability_level="middle",
        avg_mastery=0.47,
        interaction_count=50,
        correct_rate=0.5,
        interaction_density="medium",
        learning_preference="step_by_step",
    )
    p.sig_embedding = np.random.RandomState(0).randn(32).astype(np.float32)
    p.individual_embedding = np.random.RandomState(0).randn(32).astype(np.float32)
    p.fused_embedding = p.individual_embedding
    return p
