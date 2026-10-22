import pytest

from app.core import rerank


def test_rerank_off_identity():
    assert rerank.rerank("q", ["a", "b", "c"], engine="off") == [0, 1, 2]


def test_rerank_empty_candidates():
    assert rerank.rerank("q", [], engine="local") == []
    assert rerank.rerank("q", [], engine="llm", client=None, llm_model="m") == []
    assert rerank.rerank("q", [], engine="off") == []


def test_local_rerank_sorts_desc(monkeypatch):
    monkeypatch.setattr(rerank.llama_client, "rerank_pairs", lambda q, docs: [0.1, 0.9, 0.5])
    assert rerank.local_rerank("q", ["a", "b", "c"]) == [1, 2, 0]


def test_local_rerank_empty_candidates_does_not_call_service(monkeypatch):
    monkeypatch.setattr(rerank.llama_client, "rerank_pairs", lambda q, docs: pytest.fail("空候选不应调服务"))
    assert rerank.local_rerank("q", []) == []


def test_local_rerank_truncates_long_pairs(monkeypatch):
    """超长候选必须先截断再送服务。

    llama.cpp 与 torch 的 CrossEncoder 不同：它**不截断**，超长输入直接 500
    （2026-10-16 实测：≤800 字通过、≥1200 字必挂）。旧 torch 版按 512 token 截断
    （会话 28 的 max_length 档），这里对齐同一口径。
    """
    seen = {}

    def fake_rerank(query, docs):
        seen["query"] = query
        seen["docs"] = docs
        return [0.0] * len(docs)

    monkeypatch.setattr(rerank.llama_client, "rerank_pairs", fake_rerank)

    rerank.local_rerank("短问题", ["甲" * 2000])

    assert len(seen["query"]) == len("短问题")  # 短查询不被截
    assert len(seen["docs"][0]) == rerank._MAX_PAIR_CHARS - len("短问题")


def test_local_rerank_keeps_pair_within_budget(monkeypatch):
    """查询本身很长时，正文预算要相应缩短——服务的上限约束的是**整对**长度。

    实测口径：5 字查询 + 800 字正文通过、+1200 字挂。若「每段各截 512」，一个 500 字的
    长问题配上 512 字的正文就是 1024 字，照样 500——而用户会看到「检索服务暂时不可用」，
    文案还是错的（不是服务不可用，是输入太长）。
    """
    seen = {}

    def fake_rerank(query, docs):
        seen["query"] = query
        seen["docs"] = docs
        return [0.0] * len(docs)

    monkeypatch.setattr(rerank.llama_client, "rerank_pairs", fake_rerank)

    rerank.local_rerank("乙" * 2000, ["甲" * 2000])

    total = len(seen["query"]) + len(seen["docs"][0])
    assert total <= rerank._MAX_PAIR_CHARS
    assert len(seen["docs"][0]) >= rerank._MIN_DOC_CHARS  # 正文仍留得住可用信息


def test_llm_rerank_normalizes(monkeypatch):
    class _Resp:
        ranking = [2, 99, 0]  # 99 越界，1 缺失

    class _Completions:
        @staticmethod
        def create(**kwargs):
            return _Resp()

    class _IClient:
        chat = type("_Chat", (), {"completions": _Completions()})()

    monkeypatch.setattr(rerank.instructor, "from_openai", lambda client: _IClient())
    assert rerank.llm_rerank(None, "m", "q", ["a", "b", "c"]) == [2, 0, 1]


def test_llm_rerank_dedups(monkeypatch):
    class _Resp:
        ranking = [0, 1, 1]  # 重复索引 1

    class _Completions:
        @staticmethod
        def create(**kwargs):
            return _Resp()

    class _IClient:
        chat = type("_Chat", (), {"completions": _Completions()})()

    monkeypatch.setattr(rerank.instructor, "from_openai", lambda client: _IClient())
    assert rerank.llm_rerank(None, "m", "q", ["a", "b", "c"]) == [0, 1, 2]


def test_rerank_unknown_engine_raises():
    with pytest.raises(ValueError):
        rerank.rerank("q", ["a"], engine="locla")


def test_rerank_llm_without_client_degrades():
    assert rerank.rerank("q", ["a", "b"], engine="llm", client=None, llm_model="") == [0, 1]
