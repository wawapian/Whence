"""llama-server HTTP 客户端：向量嵌入 / 重排 / 探活。

两个进程各装一个模型（spec §3.1）：向量 :8094、重排 :8095，由系统独立托管。
本模块是**唯一**的出口——上层（llm / rerank）不直接碰 HTTP。

失败语义（spec §3.6）：连接失败 / 超时 / 非 2xx / 响应体损坏，一律收敛成
`LlamaUnavailableError`，由 main 的全局处理器翻成 503——**绝不静默降级**。
"""
import time

import httpx

from app.config import Settings

_settings_cache: Settings | None = None
_client_cache: httpx.Client | None = None


class LlamaUnavailableError(RuntimeError):
    """llama-server 不可用（连不上 / 超时 / 非 2xx / 响应体损坏）。"""


def _settings() -> Settings:
    """懒加载配置并缓存（Settings 每次构造都读 .env，而摄取侧会循环调用）。"""
    global _settings_cache
    if _settings_cache is None:
        _settings_cache = Settings()
    return _settings_cache


def _new_client() -> httpx.Client:
    """构造缝（同 llm._new_embedder 的先例）：测试替换 `client()` 注入 MockTransport。"""
    return httpx.Client(timeout=_settings().llama_timeout_s)


def client() -> httpx.Client:
    """懒加载、复用连接的客户端（httpx.Client 线程安全，可跨请求共享）。"""
    global _client_cache
    if _client_cache is None:
        _client_cache = _new_client()
    return _client_cache


def _post(url: str, payload: dict) -> dict:
    """POST 并解析 JSON；一切失败收敛为 `LlamaUnavailableError`。"""
    try:
        resp = client().post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
    except (httpx.HTTPError, ValueError) as exc:  # ValueError 覆盖 JSON 解析失败
        raise LlamaUnavailableError(f"llama-server 调用失败：{url}（{exc}）") from exc
    if not isinstance(data, dict):
        raise LlamaUnavailableError(f"llama-server 响应不是 JSON 对象：{url}")
    return data


def embed_texts(texts: list[str]) -> list[list[float]]:
    """文本 → 向量（按响应 `index` 复原顺序）。

    条数不符 / 缺字段一律报错：静默错位等于把别人的向量当成这条文本的向量，
    检索结果错乱且无任何症状（spec Review Focus #1/#2）。
    """
    settings = _settings()
    url = f"{settings.llama_embed_url}/v1/embeddings"
    data = _post(url, {"input": texts, "model": settings.embedding_model})
    items = data.get("data")
    if not isinstance(items, list) or len(items) != len(texts):
        got = len(items) if isinstance(items, list) else "非列表"
        raise LlamaUnavailableError(f"向量条数不符：{url} 期望 {len(texts)} 条，实际 {got}")

    out: list[list[float] | None] = [None] * len(texts)
    for item in items:
        index = item.get("index") if isinstance(item, dict) else None
        embedding = item.get("embedding") if isinstance(item, dict) else None
        if not isinstance(index, int) or not 0 <= index < len(texts) or not isinstance(embedding, list):
            raise LlamaUnavailableError(f"向量响应项缺 index/embedding：{url}（{item!r}）")
        out[index] = embedding
    if any(v is None for v in out):
        raise LlamaUnavailableError(f"向量响应的 index 未覆盖全部输入：{url}")
    return [v for v in out if v is not None]


def rerank_pairs(query: str, docs: list[str]) -> list[float]:
    """query×docs 相关性分数（与 `docs` 等长）。

    个数不符一律报错：默认 0 分会静默改变排序，症状只是「排得不太对」（spec Review Focus #3）。
    """
    settings = _settings()
    url = f"{settings.llama_rerank_url}/v1/rerank"
    data = _post(url, {"query": query, "documents": docs, "model": settings.rerank_model})
    results = data.get("results")
    if not isinstance(results, list) or len(results) != len(docs):
        got = len(results) if isinstance(results, list) else "非列表"
        raise LlamaUnavailableError(f"重排条数不符：{url} 期望 {len(docs)} 条，实际 {got}")

    scores: list[float | None] = [None] * len(docs)
    for item in results:
        index = item.get("index") if isinstance(item, dict) else None
        score = item.get("relevance_score") if isinstance(item, dict) else None
        if not isinstance(index, int) or not 0 <= index < len(docs) or not isinstance(score, (int, float)):
            raise LlamaUnavailableError(f"重排响应项缺 index/relevance_score：{url}（{item!r}）")
        scores[index] = float(score)
    if any(s is None for s in scores):
        raise LlamaUnavailableError(f"重排响应的 index 未覆盖全部候选：{url}")
    return [s for s in scores if s is not None]


def health(url: str) -> bool:
    """探活：200 且 body 为 `{"status":"ok"}` → True；其余一律 False（不抛）。"""
    try:
        resp = client().get(f"{url}/health")
        if resp.status_code != 200:
            return False
        data = resp.json()
    except (httpx.HTTPError, ValueError):
        return False
    return isinstance(data, dict) and data.get("status") == "ok"


def wait_ready(url: str, timeout_s: float, interval_s: float = 1.0) -> bool:
    """轮询 `/health` 直到就绪或超时；超时返回 False，不抛（预热失败只 warning）。"""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if health(url):
            return True
        time.sleep(interval_s)
    return False
