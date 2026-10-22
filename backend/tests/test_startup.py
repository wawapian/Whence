"""启动提速：重库惰性导入钉子 + 加载锁单次构造 + 启动预热语义。"""
import asyncio
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from app import main as main_module
from app.config import Settings
from app.core import llama_client

BACKEND_DIR = Path(__file__).resolve().parent.parent


# --- 惰性导入 -------------------------------------------------------------

def test_import_app_main_keeps_heavy_stack_out():
    """冷启动（import app.main）不得把 sentence_transformers/torch/transformers 拖进内存。"""
    code = (
        "import sys, app.main; "
        "assert 'sentence_transformers' not in sys.modules, 'sentence_transformers 进了导入链'; "
        "assert 'torch' not in sys.modules, 'torch 进了导入链'; "
        "assert 'transformers' not in sys.modules, 'transformers 进了导入链'"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=BACKEND_DIR, capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, proc.stderr


# --- 加载锁用例：随进程内构造一并删除 ---------------------------------------
# 原 test_concurrent_first_access_constructs_once 赌的是 llm._get_embedder /
# rerank._get_cross 的双检锁。换引擎后模型住在 llama-server 进程里，本地已无构造可锁，
# 两个函数也随 spec §3.3 删除——**此处是主动删除，不是漏测**。


# --- 预热语义 -------------------------------------------------------------

def test_warmup_probes_then_sends_minimal_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(llama_client, "wait_ready", lambda url, t: calls.append(("ready", url)) or True)
    monkeypatch.setattr(llama_client, "embed_texts", lambda texts: calls.append(("embed", texts)) or [[0.0]])
    monkeypatch.setattr(llama_client, "rerank_pairs", lambda q, docs: calls.append(("rerank", q, docs)) or [0.0])
    settings = Settings(_env_file=None, rerank_engine="local")

    main_module._warmup_models(settings)

    assert calls == [
        ("ready", settings.llama_embed_url),
        ("embed", ["warmup"]),
        ("ready", settings.llama_rerank_url),
        ("rerank", "warmup", ["warmup"]),
    ]


def test_warmup_skips_rerank_when_engine_not_local(monkeypatch):
    calls = []
    monkeypatch.setattr(llama_client, "wait_ready", lambda url, t: calls.append(url) or True)
    monkeypatch.setattr(llama_client, "embed_texts", lambda texts: [[0.0]])
    monkeypatch.setattr(llama_client, "rerank_pairs", lambda q, docs: pytest.fail("工程非 local 不应调重排"))
    settings = Settings(_env_file=None, rerank_engine="off")

    main_module._warmup_models(settings)

    assert calls == [settings.llama_embed_url]


def test_warmup_swallows_failures_and_continues(monkeypatch):
    attempted = []
    monkeypatch.setattr(llama_client, "wait_ready", lambda url, t: attempted.append(("ready", url)) or True)

    def boom(texts):
        attempted.append(("embed", texts))
        raise llama_client.LlamaUnavailableError("服务挂")

    monkeypatch.setattr(llama_client, "embed_texts", boom)
    settings = Settings(_env_file=None, rerank_engine="local")

    main_module._warmup_models(settings)  # 不抛

    assert attempted[0] == ("ready", settings.llama_embed_url)
    assert attempted[1][0] == "embed"


# --- lifespan 接线：后台、不阻塞、可关 --------------------------------------

class _NullSession:
    def close(self):
        pass


def _stub_lifespan(monkeypatch, *, warmup):
    """把 lifespan 的启动扫尾换成替身（免打真实 data_dir），只留预热接线可测。"""
    settings = Settings(_env_file=None, warmup=warmup)
    monkeypatch.setattr(main_module.deps, "get_settings", lambda: settings)
    monkeypatch.setattr(main_module.deps, "_session_factory", lambda s: lambda: _NullSession())
    monkeypatch.setattr(main_module, "sweep_expired", lambda session: None)
    monkeypatch.setattr(main_module, "requeue_pending", lambda session, s: (0, 0))
    return settings


def test_lifespan_runs_warmup_as_background_daemon(monkeypatch):
    """预热线程要起得来、是 daemon，且不在启动关键路径上（lifespan 不等它）。"""
    expected = _stub_lifespan(monkeypatch, warmup=True)
    started = threading.Event()
    release = threading.Event()
    seen = {}

    def blocking_warmup(settings):
        seen["daemon"] = threading.current_thread().daemon
        seen["model"] = settings.embedding_model
        started.set()
        release.wait(timeout=10)

    monkeypatch.setattr(main_module, "_warmup_models", blocking_warmup)

    async def body():
        t0 = time.perf_counter()
        async with main_module.lifespan(main_module.app):
            assert started.wait(timeout=5), "预热线程没起来"
            assert seen["daemon"] is True, "预热线程不是 daemon"
            assert time.perf_counter() - t0 < 5, "lifespan 阻塞等待了预热线程"
            assert seen["model"] == expected.embedding_model

    try:
        asyncio.run(body())
    finally:
        release.set()


def test_lifespan_skips_warmup_when_disabled(monkeypatch):
    _stub_lifespan(monkeypatch, warmup=False)
    called = []
    monkeypatch.setattr(main_module, "_warmup_models", lambda settings: called.append(settings))

    async def body():
        async with main_module.lifespan(main_module.app):
            pass

    asyncio.run(body())
    time.sleep(0.1)
    assert called == []


# --- 配置 ----------------------------------------------------------------

def test_warmup_defaults_on(monkeypatch):
    monkeypatch.delenv("WARMUP", raising=False)
    assert Settings(_env_file=None).warmup is True


def test_warmup_can_be_disabled_by_env(monkeypatch):
    monkeypatch.setenv("WARMUP", "false")
    assert Settings(_env_file=None).warmup is False


def test_conftest_disables_warmup_for_tests():
    """conftest 兜底：测试环境默认关闭预热，防未来 `with TestClient(...)` 误触真加载。"""
    assert Settings(_env_file=None).warmup is False
