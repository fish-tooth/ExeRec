"""测试 LLM Factory 和 Mock 后端"""
import pytest


def test_mock_chat(mock_llm):
    resp = mock_llm.chat([
        {"role": "user", "content": "请生成一段反思 JSON"}
    ])
    assert resp.content
    assert resp.backend == "mock"


def test_mock_reflection_format(mock_llm):
    """mock 应该对包含'反思'的 prompt 返回结构化 JSON"""
    import json
    resp = mock_llm.chat([
        {"role": "user", "content": "请生成反思 lesson 和 action_delta"}
    ])
    data = json.loads(resp.content)
    assert "lesson" in data
    assert "action_delta" in data


def test_cache_enable(mock_cfg, tmp_path):
    """同样的请求应命中缓存"""
    from llm_adapter import LLMFactory
    mock_cfg["llm"]["cache"]["enable"] = True
    mock_cfg["llm"]["cache"]["cache_dir"] = str(tmp_path / "cache")
    
    llm = LLMFactory.from_config(mock_cfg)
    msgs = [{"role": "user", "content": "测试同一个 prompt"}]
    
    resp1 = llm.chat(msgs)
    resp2 = llm.chat(msgs)
    
    assert resp1.content == resp2.content
    # 第二次应该命中缓存
    assert resp2.cached is True
    assert llm.cache_hit >= 1


def test_stats(mock_llm):
    msgs = [{"role": "user", "content": "hello"}]
    mock_llm.chat(msgs)
    s = mock_llm.stats()
    assert s["backend"] == "mock"
    assert s["call_count"] >= 1
