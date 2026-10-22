import logging
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app import deps
from app.api import documents, export, files, memories, model_settings, qa_history, query, reminders, trash, workspace
from app.core import model_preferences
from app.config import Settings
from app.core import llama_client
from app.core.recovery import requeue_pending
from app.core.trash import sweep_expired

FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"

logger = logging.getLogger(__name__)


_WARMUP_TEXT = "warmup"


def _warmup_models(settings: Settings) -> None:
    """后台预热：等两个 llama-server 就绪，并各发一次最小请求（让它们把模型读进内存）。

    模型现在住在**外部进程**里（spec §3.1），所以这里的动作是「探活 + 最小调用」而非
    「加载权重」。各步失败只记 warning，绝不抛——服务没起来不该拖住应用启动。
    """
    done = []
    t0 = time.perf_counter()
    try:
        if llama_client.wait_ready(settings.llama_embed_url, settings.llama_warmup_timeout_s):
            llama_client.embed_texts([_WARMUP_TEXT])
            done.append(f"embedding={settings.embedding_model} {time.perf_counter() - t0:.1f}s")
        else:
            logger.warning("启动预热：向量服务未就绪（%s）", settings.llama_embed_url)
    except Exception:
        logger.warning("启动预热 embedding 失败（首问时将重试）", exc_info=True)
    if settings.rerank_engine == "local":
        t1 = time.perf_counter()
        try:
            if llama_client.wait_ready(settings.llama_rerank_url, settings.llama_warmup_timeout_s):
                llama_client.rerank_pairs(_WARMUP_TEXT, [_WARMUP_TEXT])
                done.append(f"rerank={settings.rerank_model} {time.perf_counter() - t1:.1f}s")
            else:
                logger.warning("启动预热：重排服务未就绪（%s）", settings.llama_rerank_url)
        except Exception:
            logger.warning("启动预热 rerank 失败（首问时将重试）", exc_info=True)
    logger.info("启动预热结束：%s", "；".join(done) if done else "无（见上方 warning）")


# 注意：deps.get_settings() 不走 dependency_overrides——测试勿用 `with TestClient(...)`
# （进入上下文会触发 lifespan），否则启动清扫/重投会打到真实 data_dir。
@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时：清一次过期的回收站对象 + 重投重启前丢失的后台任务，并起后台线程预热模型；
    后台预热不 join（不拖慢启动）、失败只记日志；扫尾失败也只记 warning，不阻塞启动。"""
    try:
        settings = deps.get_settings()
        if settings.warmup:
            # daemon：进程退出时随进程结束，不为它多等；与下面的扫尾并行跑
            threading.Thread(
                target=_warmup_models, args=(settings,), name="warmup-models", daemon=True
            ).start()
        session = deps._session_factory(settings)()
        try:
            sweep_expired(session)
            n_records, n_docs = requeue_pending(session, settings)
            if n_records or n_docs:
                logger.warning("启动重投：%d 条记录整理、%d 份文档解析", n_records, n_docs)
        finally:
            session.close()
    except Exception:
        logger.warning("启动扫尾失败，已跳过", exc_info=True)
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Whence · 来处", lifespan=lifespan)

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True}

    @app.exception_handler(llama_client.LlamaUnavailableError)
    async def _llama_unavailable(request, exc):
        """本地推理服务不可用 → **503 + 明确文案**（全端点统一出口，spec §3.6）。

        严禁在这里改成「没有找到相关记录」之类的降级文案——检索失败被说成「库里没有」，
        是另一种形式的编造。
        """
        return JSONResponse(status_code=503, content={"detail": "检索服务暂时不可用，请稍后再试"})

    app.include_router(memories.router)
    app.include_router(query.router)
    app.include_router(reminders.router)
    app.include_router(export.router)
    app.include_router(files.router)
    app.include_router(documents.router)
    app.include_router(qa_history.router)
    # 必须在下面的 StaticFiles 挂载之前：否则 /api/trash 被静态托管吃掉
    app.include_router(trash.router)
    app.include_router(workspace.router)
    app.include_router(model_settings.router)

    if FRONTEND_DIR.exists():
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

    return app


app = create_app()
