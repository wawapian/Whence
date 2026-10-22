from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def test_settings_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    s = Settings(_env_file=None)
    assert s.llm_api_key == "sk-test"
    assert s.data_dir == tmp_path
    assert s.embedding_model == "BAAI/bge-m3"


def test_settings_defaults():
    s = Settings(_env_file=None)
    assert s.llm_provider == "deepseek"
    assert s.llm_base_url == "https://api.deepseek.com"
    assert s.llm_model == "deepseek-flash"
    assert s.embedding_model == "BAAI/bge-m3"
    assert s.data_dir == Path("./data")
    assert s.rerank_engine == "local"
    assert s.rerank_model == "BAAI/bge-reranker-base"


def test_provider_preset_kimi(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "kimi")
    s = Settings()
    assert s.llm_base_url == "https://api.moonshot.cn/v1"
    assert s.llm_model == "kimi-k3"


def test_provider_preset_override(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "kimi")
    monkeypatch.setenv("LLM_BASE_URL", "https://custom.example.com")
    s = Settings()
    assert s.llm_base_url == "https://custom.example.com"  # 显式覆盖预设


def test_health_check():
    client = TestClient(create_app())
    assert client.get("/healthz").status_code == 200
    assert client.get("/healthz").json() == {"ok": True}


def test_llama_settings_defaults():
    s = Settings(_env_file=None)
    assert s.llama_embed_url == "http://127.0.0.1:8094"
    assert s.llama_rerank_url == "http://127.0.0.1:8095"
    assert s.llama_timeout_s == 30.0
    assert s.llama_warmup_timeout_s == 60.0


def test_llama_embed_url_from_env(monkeypatch):
    monkeypatch.setenv("LLAMA_EMBED_URL", "http://10.0.0.5:9000")
    assert Settings(_env_file=None).llama_embed_url == "http://10.0.0.5:9000"
