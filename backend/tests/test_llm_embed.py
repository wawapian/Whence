"""llm.embed 换引擎后的委派语义与 embedding_dim 的纯查表语义。"""
import pytest

from app.core import llm, llama_client


def test_embed_delegates_single_text(monkeypatch):
    seen = []
    monkeypatch.setattr(llama_client, "embed_texts", lambda texts: seen.append(texts) or [[0.5, 0.5]])
    assert llm.embed("你好") == [0.5, 0.5]
    assert seen == [["你好"]]


def test_embedding_dim_known_models_do_not_touch_network(monkeypatch):
    monkeypatch.setattr(llama_client, "embed_texts", lambda texts: pytest.fail("查表不应触网"))
    assert llm.embedding_dim("BAAI/bge-m3") == 1024
    assert llm.embedding_dim("BAAI/bge-small-zh-v1.5") == 512


def test_embedding_dim_unknown_model_raises_with_hint():
    with pytest.raises(ValueError, match="登记"):
        llm.embedding_dim("nope/nope")
