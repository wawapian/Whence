import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi import Depends
from openai import OpenAI

from app.config import Settings
from app.core import model_preferences, qa
from app.deps import get_settings


@pytest.mark.parametrize("provider,model,temperature,field,extra", [
    ("deepseek", "deepseek-flash", 0.3, "max_tokens", {"thinking": {"type": "disabled"}}),
    ("kimi", "kimi-k3", None, "max_completion_tokens", {}),
    ("kimi", "kimi-k2.6", None, "max_tokens", {"thinking": {"type": "disabled"}}),
    ("qwen", "qwen3.8-flash", 0.3, "max_tokens", {"enable_thinking": False}),
])
def test_actual_provider_payload(provider, model, temperature, field, extra):
    seen = []
    def transport(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"id": "x", "object": "chat.completion", "created": 0,
            "model": model, "choices": [{"index": 0, "finish_reason": "stop",
                                         "message": {"role": "assistant", "content": "ok"}}]})
    settings = Settings(_env_file=None, llm_provider=provider, llm_model=model,
                        llm_temperature=0.3, llm_max_tokens=512)
    client = model_preferences.configure_client(OpenAI(base_url=settings.llm_base_url, api_key="test",
        http_client=httpx.Client(transport=httpx.MockTransport(transport))), settings)
    with client:
        client.with_options(timeout=90).chat.completions.create(model=model, messages=[], max_tokens=1024)
    payload = seen[0]
    assert payload[field] == 512
    assert ("max_completion_tokens" if field == "max_tokens" else "max_tokens") not in payload
    assert payload.get("temperature") == temperature
    for key, value in extra.items():
        assert payload[key] == value






def test_thinking_tool_turn_preserves_full_assistant_message(monkeypatch):
    tool = SimpleNamespace(id="call_1", function=SimpleNamespace(name="test", arguments="{}"))
    raw = {"role": "assistant", "content": "", "reasoning_content": "opaque reasoning",
           "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "test", "arguments": "{}"}}]}
    msg = SimpleNamespace(content="", tool_calls=[tool], model_dump=lambda **_: raw)
    seen = []
    def create(**kwargs):
        seen.append(list(kwargs["messages"]))
        current = msg if len(seen) == 1 else SimpleNamespace(content="answer", tool_calls=[])
        return SimpleNamespace(choices=[SimpleNamespace(message=current)])
    monkeypatch.setattr(qa.toolbox, "run_tool", lambda *args: "result")
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    assert qa._loop(client, Settings(_env_file=None), [], None) == "answer"
    assert seen[1][0] == raw
