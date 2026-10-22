"""llama_client：HTTP 客户端行为与失败语义（全部走 MockTransport，不打真服务）。"""
import httpx
import pytest

from app.core import llama_client


def _stub(monkeypatch, handler):
    monkeypatch.setattr(
        llama_client, "client", lambda: httpx.Client(transport=httpx.MockTransport(handler))
    )


def test_embed_texts_restores_index_order(monkeypatch):
    def handler(request):
        assert request.url.path == "/v1/embeddings"
        return httpx.Response(200, json={"data": [
            {"index": 1, "embedding": [0.0, 1.0]},
            {"index": 0, "embedding": [1.0, 0.0]},
        ]})
    _stub(monkeypatch, handler)
    assert llama_client.embed_texts(["a", "b"]) == [[1.0, 0.0], [0.0, 1.0]]


def test_embed_texts_count_mismatch_raises(monkeypatch):
    _stub(monkeypatch, lambda r: httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]}))
    with pytest.raises(llama_client.LlamaUnavailableError):
        llama_client.embed_texts(["a", "b"])


def test_embed_texts_missing_embedding_field_raises(monkeypatch):
    _stub(monkeypatch, lambda r: httpx.Response(200, json={"data": [{"index": 0}, {"index": 1}]}))
    with pytest.raises(llama_client.LlamaUnavailableError):
        llama_client.embed_texts(["a", "b"])


def test_embed_connection_error_becomes_unavailable(monkeypatch):
    def handler(request):
        raise httpx.ConnectError("connection refused")
    _stub(monkeypatch, handler)
    with pytest.raises(llama_client.LlamaUnavailableError):
        llama_client.embed_texts(["a"])


def test_embed_non_json_body_becomes_unavailable(monkeypatch):
    _stub(monkeypatch, lambda r: httpx.Response(200, content=b"<html>not json</html>"))
    with pytest.raises(llama_client.LlamaUnavailableError):
        llama_client.embed_texts(["a"])


def test_rerank_pairs_returns_scores_in_docs_order(monkeypatch):
    def handler(request):
        assert request.url.path == "/v1/rerank"
        return httpx.Response(200, json={"results": [
            {"index": 2, "relevance_score": 0.5},
            {"index": 0, "relevance_score": 0.2},
            {"index": 1, "relevance_score": 0.9},
        ]})
    _stub(monkeypatch, handler)
    assert llama_client.rerank_pairs("q", ["a", "b", "c"]) == [0.2, 0.9, 0.5]


def test_rerank_count_mismatch_raises(monkeypatch):
    _stub(monkeypatch, lambda r: httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.2}]}))
    with pytest.raises(llama_client.LlamaUnavailableError):
        llama_client.rerank_pairs("q", ["a", "b"])


def test_health_true_on_ok_false_on_error(monkeypatch):
    _stub(monkeypatch, lambda r: httpx.Response(200, json={"status": "ok"}))
    assert llama_client.health("http://127.0.0.1:8094") is True
    _stub(monkeypatch, lambda r: httpx.Response(503, json={"error": "loading"}))
    assert llama_client.health("http://127.0.0.1:8094") is False


class _FakeClock:
    def __init__(self):
        self.t = 0.0

    def monotonic(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


def test_wait_ready_polls_until_ok(monkeypatch):
    clock = _FakeClock()
    calls = []
    monkeypatch.setattr(llama_client, "time", clock)
    monkeypatch.setattr(llama_client, "health", lambda url: calls.append(url) or len(calls) >= 3)
    assert llama_client.wait_ready("http://x", timeout_s=60) is True
    assert len(calls) == 3


def test_wait_ready_gives_up_after_timeout(monkeypatch):
    clock = _FakeClock()
    monkeypatch.setattr(llama_client, "time", clock)
    monkeypatch.setattr(llama_client, "health", lambda url: False)
    assert llama_client.wait_ready("http://x", timeout_s=10, interval_s=4) is False
    assert clock.t >= 10  # 确实等满了 timeout 才放弃
