from pydantic import ValidationError

from app.core import extraction
from app.models.schemas import EVENT_TYPES, STATE_TYPES, ExtractedFact, ExtractionResult


def test_state_types_expanded():
    assert set(STATE_TYPES) >= {
        "item_location", "residence", "contact", "employer", "bank_card",
        "health", "preference", "relationship", "account", "vehicle", "pet",
    }


def test_event_types():
    assert set(EVENT_TYPES) >= {
        "consumption", "trip", "todo", "schedule", "meeting", "idea", "other",
    }


def test_extracted_fact_defaults():
    f = ExtractedFact(type="item_location", summary="身份证放书房", confidence=0.8)
    assert f.people == []
    assert f.items == []
    assert f.is_reminder is False
    assert f.subject_key is None


def test_extracted_fact_confidence_range():
    try:
        ExtractedFact(type="item_location", summary="x", confidence=1.5)
        assert False, "confidence 超出 0~1 应校验失败"
    except ValidationError:
        pass


def _fake_instructor(monkeypatch, response):
    """instructor 替身：create 记录每次 kwargs，按 response 返回。"""
    calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return response

    class FakeIClient:
        def __init__(self):
            self.chat = type("Chat", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr(extraction.instructor, "from_openai", lambda c: FakeIClient())
    return calls


def test_extract_returns_all_facts(monkeypatch):
    """extract 返回**全部事实**（list，保持模型输出顺序），内部用 ExtractionResult 包装模型。"""
    expected = [
        ExtractedFact(type="item_location", subject_key="耳机:位置",
                      summary="耳机放到书柜里了", confidence=0.9),
        ExtractedFact(type="item_location", subject_key="耳机盒子:位置",
                      summary="耳机盒子放到进门左边立柜里了", confidence=0.9),
    ]
    calls = _fake_instructor(monkeypatch, ExtractionResult(facts=expected))

    facts = extraction.extract(object(), "m", "耳机放书柜里了，耳机盒子放到进门左边立柜里了")

    assert facts == expected
    assert calls[0]["response_model"] is ExtractionResult  # 单对象 response_model 已换包装模型


def test_extract_prompt_splits_all_facts_and_keeps_states(monkeypatch):
    """prompt 硬约束钉住（发现#2-①）：拆全部独立事实、一句话可多事实、状态类不遗漏、
    按出现顺序、每条遵循 subject_key 规则。"""
    calls = _fake_instructor(monkeypatch, ExtractionResult(facts=[]))

    extraction.extract(object(), "m", "文本")

    prompt = calls[0]["messages"][0]["content"]
    assert "全部独立事实" in prompt
    assert "多个事实" in prompt
    assert "不得遗漏" in prompt
    assert "顺序" in prompt
    assert "subject_key" in prompt


def test_extract_empty_facts_stays_empty(monkeypatch):
    """模型零事实 → 返回空列表（pending 兜底由调用方决定，extract 本身不造事实）。"""
    _fake_instructor(monkeypatch, ExtractionResult(facts=[]))

    assert extraction.extract(object(), "m", "文本") == []
