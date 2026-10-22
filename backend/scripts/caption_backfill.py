"""存量图片 caption 补录脚本：给旧文档中还没有 caption 的图片补生成描述并建索引。

用法：在 backend 目录执行 `python scripts/caption_backfill.py`（配置读 `Settings()`/.env）

行为：遍历 `images.caption IS NULL` 且所属 Document `allow_cloud != 0` 的图片，逐图：
读 `data_dir / file_path` → VLM 中性描述（提示词与入库一致）→ 写 `images.caption`
→ 建「图片描述（《标题》…）：caption」chunk 并索引（text/seq 与入库一致）→ 逐图提交。
单图缺失/描述失败/空描述只记 failed 跳过，绝不影响其他图片。未配置 `vlm_model`
时不发任何请求，直接返回可补录数量。
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.files import MAX_CAPTIONS_PER_DOC, _NEUTRAL_CAPTION_PROMPT
from app.config import Settings
from app.core import cloud, llm
from app.core.answer import image_source
from app.deps import _session_factory
from app.memory import chunks, documents
from app.models.db import Chunk, Document, Image

logger = logging.getLogger(__name__)


def _pending_images(session: Session) -> list[Image]:
    """待补录图片：caption 为空、所属文档允许上云（与入库双门控同侧），且非疑似装饰小图。"""
    return session.scalars(
        select(Image)
        .join(Document)
        .where(
            Image.caption.is_(None),
            Document.allow_cloud != 0,
            Image.decorative == 0,
        )
    ).all()


def backfill(settings: Settings, session: Session | None = None) -> dict:
    """补录存量图片 caption，返回 `{"captioned": n, "failed": m, "remaining": k}`。

    session 为 None 时按 settings 建独立会话；注入时直接复用（测试）。
    每轮每文档最多处理 MAX_CAPTIONS_PER_DOC 张，超出的留 NULL、下次运行继续补。
    """
    own_session = session is None
    if own_session:
        session = _session_factory(settings)()
    try:
        pending = _pending_images(session)
        if not settings.vlm_model:
            return {"captioned": 0, "failed": 0, "remaining": len(pending)}
        client = cloud.build_vlm_client(settings)
        captioned = failed = 0
        processed_per_doc: dict[str, int] = {}
        for image in pending:
            # 回滚会过期 ORM 实例，日志/异常处理只读进循环前绑定的值，避免二次抛错
            image_id = image.id
            doc_id = image.document_id
            processed_per_doc[doc_id] = processed_per_doc.get(doc_id, 0) + 1
            if processed_per_doc[doc_id] > MAX_CAPTIONS_PER_DOC:
                continue  # 超出每文档上限：留 NULL 给下一轮
            try:
                if not image.file_path:
                    failed += 1
                    logger.warning("image %s has no file_path, skipped", image_id)
                    continue
                path = Path(settings.data_dir) / image.file_path
                if not path.is_file():
                    failed += 1
                    logger.warning("image %s file missing: %s", image_id, path)
                    continue
                caption = cloud.describe_image(
                    client, settings.vlm_model, path.read_bytes(), _NEUTRAL_CAPTION_PROMPT
                )
                if not caption:
                    failed += 1
                    logger.warning("image %s caption empty, skipped", image_id)
                    continue
                image.caption = caption
                doc = session.get(Document, doc_id)
                text = f"图片描述（{image_source(doc, image)}）：{caption}"
                max_seq = session.scalar(
                    select(func.max(Chunk.seq)).where(Chunk.document_id == doc_id)
                )
                chunk = documents.insert_chunk(
                    session,
                    document_id=doc_id,
                    seq=0 if max_seq is None else max_seq + 1,
                    text=text,
                    heading_path="图片",
                    parent_text=text,
                )
                chunks.index_chunk(
                    session, chunk.id, llm.embed(text, model=settings.embedding_model), text
                )
                session.commit()
                captioned += 1
            except Exception:
                # 只丢弃该图的部分写入；此前图片均已逐图提交，后面的图继续处理
                failed += 1
                try:
                    session.rollback()
                    logger.warning("image %s caption failed", image_id, exc_info=True)
                except Exception:
                    # 连接已坏时 rollback 也可能抛错——绝不能逃出循环
                    logger.warning("image %s caption failed; rollback also errored", image_id)
        return {"captioned": captioned, "failed": failed, "remaining": len(_pending_images(session))}
    finally:
        if own_session:
            session.close()


if __name__ == "__main__":
    print(backfill(Settings()))
