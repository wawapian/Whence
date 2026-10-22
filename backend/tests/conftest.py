"""全局测试夹具：默认禁用真实 LLM 客户端。"""
import pytest

from app.core import llm


@pytest.fixture(autouse=True)
def _disable_warmup(monkeypatch):
    """测试环境一律关掉启动预热：未来若有测试 `with TestClient(...)` 进 lifespan，
    也不会在后台拉起真实模型加载。需要断言默认值的用例自行 delenv 覆盖。"""
    monkeypatch.setenv("WARMUP", "false")


@pytest.fixture(autouse=True)
def _disable_llm_client(monkeypatch):
    """所有内部 LLM 调用（意图判定/查询扩展/合成/云总结）在测试中一律失败走降级路径，
    避免误连真实 API。需要真实客户端行为的测试自行 monkeypatch 覆盖即可。"""

    class _DeadCompletions:
        @staticmethod
        def create(**kwargs):
            raise RuntimeError("tests: LLM disabled")

    class _DeadClient:
        chat = type("Chat", (), {"completions": _DeadCompletions()})()

    monkeypatch.setattr(llm, "build_client", lambda settings: _DeadClient())
