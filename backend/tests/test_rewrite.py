"""查询改写：多查询扩展 + 分级触发。"""
from app.config import Settings
from app.core import rewrite


class FakeClient:
    """伪造 OpenAI 客户端：create 返回固定 content（或抛异常），记录调用。"""

    def __init__(self, content=None, error=None):
        self.content = content
        self.error = error
        self.calls = []

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        message = type("M", (), {"content": self.content})()
        return type("R", (), {"choices": [type("C", (), {"message": message})()]})()


LONG_Q = "帮我看看之前那个实验里参数怎么设置的"


def _settings(**kw):
    return Settings(**{"query_rewrite": "auto", "query_rewrite_min_chars": 5, **kw})


def test_expand_queries_parses_multiline_variants():
    client = FakeClient(content="ROS 依赖配置检查\n实验四 catkin_package 配置")
    assert rewrite.expand_queries(client, "m", LONG_Q) == [
        "ROS 依赖配置检查",
        "实验四 catkin_package 配置",
    ]


def test_expand_queries_strips_numbering_and_bullets():
    client = FakeClient(content="1. 变体A\n- 变体B\n2）变体C")
    assert rewrite.expand_queries(client, "m", LONG_Q) == ["变体A", "变体B", "变体C"]


def test_expand_queries_dedupes_and_caps_at_three():
    client = FakeClient(content="变体A\n变体B\n变体A\n变体C\n变体D")
    assert rewrite.expand_queries(client, "m", LONG_Q) == ["变体A", "变体B", "变体C"]


def test_expand_queries_llm_failure_returns_empty():
    client = FakeClient(error=RuntimeError("api down"))
    assert rewrite.expand_queries(client, "m", LONG_Q) == []


def test_expand_queries_unparsable_output_returns_empty():
    client = FakeClient(content="\n\n  ")
    assert rewrite.expand_queries(client, "m", LONG_Q) == []


def test_should_expand_requires_auto_mode_and_min_length():
    assert rewrite.should_expand("查重报告", _settings()) is False  # 4 字 < 5
    assert rewrite.should_expand("查重报告在哪里放", _settings()) is True  # 8 字 ≥ 5
    assert rewrite.should_expand(LONG_Q, Settings(query_rewrite="off")) is False


def test_retrieval_queries_short_question_returns_original_only():
    client = FakeClient(content="不该被调用")
    assert rewrite.retrieval_queries("查重报告", _settings(), client) == ["查重报告"]
    assert client.calls == []


def test_retrieval_queries_expands_long_question_with_original_first():
    client = FakeClient(content="参数设置\n实验参数")
    qs = rewrite.retrieval_queries(LONG_Q, _settings(), client)
    assert qs == [LONG_Q, "参数设置", "实验参数"]
    assert client.calls  # 触发了一次 LLM 调用


def test_retrieval_queries_falls_back_to_original_on_failure():
    client = FakeClient(error=RuntimeError("api down"))
    assert rewrite.retrieval_queries(LONG_Q, _settings(), client) == [LONG_Q]
