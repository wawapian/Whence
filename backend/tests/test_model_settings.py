import json

import httpx
import pytest
from fastapi import Depends
from fastapi.testclient import TestClient
from openai import OpenAI

from app import deps
from app.config import Settings
from app.core import model_preferences, cloud
from app.main import create_app












def test_local_mode_can_save_custom_endpoint_without_editing_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "local"))
    client = TestClient(create_app())
    saved = client.put("/api/model-settings", json={"llm_provider": "custom", "llm_base_url": "http://127.0.0.1:9000/v1",
                                                  "llm_model": "local-model", "llm_api_key": "local-test-key"})
    assert saved.status_code == 200
    assert saved.json()["scope"] == "local"
    assert saved.json()["llm_base_url"] == "http://127.0.0.1:9000/v1"
    assert "local-test-key" not in saved.text
    assert client.get("/api/model-settings").json()["llm_model"] == "local-model"
    assert "no-store" in saved.headers["cache-control"]
    assert client.put("/api/model-settings", json={"llm_model": "other"}, headers={"Origin": "https://evil.invalid"}).status_code == 403






def test_generation_parameters_reach_sdk_and_survive_with_options():
    seen = []
    timeouts = []
    def transport(request):
        seen.append(json.loads(request.content))
        timeouts.append(request.extensions["timeout"]["read"])
        return httpx.Response(200, json={"id": "test", "object": "chat.completion", "created": 0,
                                         "model": "test", "choices": [{"index": 0, "finish_reason": "stop",
                                         "message": {"role": "assistant", "content": "ok"}}]})
    settings = Settings(_env_file=None, llm_temperature=0.3, llm_max_tokens=512, llm_timeout_s=7)
    client = model_preferences.configure_client(OpenAI(api_key="test", base_url=settings.llm_base_url, http_client=httpx.Client(
        transport=httpx.MockTransport(transport))), settings)
    try:
        client.with_options(timeout=90).chat.completions.create(model="test", messages=[])
        client.chat.completions.create(model="test", messages=[], temperature=0.8, max_tokens=256)
        assert seen[0]["temperature"] == 0.3 and seen[0]["max_tokens"] == 512
        assert seen[1]["temperature"] == 0.8 and seen[1]["max_tokens"] == 256
        assert timeouts[0] == 7
    finally:
        client.close()


def test_vlm_never_reuses_llm_key_for_a_different_service():
    settings = Settings(_env_file=None, llm_provider="qwen", llm_api_key="server-key",
                        vlm_base_url="https://api.openai.com/v1")
    assert not cloud.vlm_key(settings)
    settings.vlm_base_url = settings.llm_base_url
    assert cloud.vlm_key(settings) == "server-key"


def test_configured_client_supports_structured_extraction():
    from app.core import extraction
    seen = []
    def transport(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"id": "test", "object": "chat.completion", "created": 0,
            "model": "test", "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": None, "tool_calls": [{"id": "call_test", "type": "function",
                "function": {"name": "ExtractionResult", "arguments": '{"facts":[]}'}}]}}]})
    settings = Settings(_env_file=None, llm_temperature=0.2, llm_max_tokens=512)
    client = model_preferences.configure_client(OpenAI(api_key="test", base_url=settings.llm_base_url, http_client=httpx.Client(
        transport=httpx.MockTransport(transport))), settings)
    try:
        assert extraction.extract(client, "test", "测试记录") == []
        assert seen[0]["temperature"] == 0.2 and seen[0]["max_tokens"] == 512
    finally:
        client.close()
