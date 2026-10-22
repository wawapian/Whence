"""查询路由：LLM 意图判定 + 精确规则兜底。"""
from app.core import router
from app.core.router import QueryKind


class FakeClient:
    """伪造 OpenAI 客户端：返回固定 content 或抛异常。"""

    def __init__(self, content=None, error=None):
        self.content = content
        self.error = error

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        if self.error is not None:
            raise self.error
        message = type("M", (), {"content": self.content})()
        return type("R", (), {"choices": [type("C", (), {"message": message})()]})()


def test_classify_maps_llm_output_to_kind():
    assert router.classify("随便问", FakeClient(content="summarize"), "m") == QueryKind.summarize
    assert router.classify("随便问", FakeClient(content="image_query"), "m") == QueryKind.image_query
    assert router.classify("随便问", FakeClient(content="qa"), "m") == QueryKind.qa


def test_classify_parses_token_inside_text():
    assert router.classify("随便问", FakeClient(content="类别是 image_query。"), "m") == QueryKind.image_query


def test_classify_falls_back_to_rules_on_llm_failure():
    assert router.classify("总结这篇论文", FakeClient(error=RuntimeError("down")), "m") == QueryKind.summarize
    assert router.classify("身份证在哪", FakeClient(error=RuntimeError("down")), "m") == QueryKind.qa


def test_classify_falls_back_when_output_unparsable():
    assert router.classify("总结这篇论文", FakeClient(content="我不确定"), "m") == QueryKind.summarize


def test_route_rules_precise_patterns():
    """规则兜底：精确模式——不再出现「是什么」这种宽词误伤。"""
    assert router.route("总结这篇论文") == QueryKind.summarize
    assert router.route("这份报告讲了什么") == QueryKind.summarize
    assert router.route("这张图有几个颜色") == QueryKind.image_query
    assert router.route("分别是什么颜色") == QueryKind.qa  # 原 bug：被裸「是什么」误判为总结
    assert router.route("身份证在哪") == QueryKind.qa
    assert router.route("这个月我花了多少钱") == QueryKind.qa  # 统计意图为 LLM 专属：规则兜底不识别，落默认问答（降级）


def test_mentions_images():
    assert router.mentions_images("这张图里有几个颜色") is True
    assert router.mentions_images("总结这篇论文") is False


def test_classify_maps_analysis():
    assert router.classify("随便问", FakeClient(content="analysis_query"), "m") == QueryKind.analysis_query
