"""启动扫尾：重投重启前丢失的后台任务（车道是内存队列，不持久）。

- 记录整理：`raw_evidence.consolidated=0 且未软删` → 重投 record 车道
  （accept-fast 后进程退出的兜底；NULL=存量旧数据，不回填不重投）；
- 文档解析：`status='processing' 且未软删` → 重投 doc 车道
  （启动时必然为滞留——没有任何工人会在跑）。
"""
import logging

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.files import process_document
from app.api.memories import consolidate_record
from app.config import Settings
from app.core import queue
from app.models.db import Document, RawEvidence

logger = logging.getLogger(__name__)


def requeue_pending(session: Session, settings: Settings) -> tuple[int, int]:
    """重投滞留任务，返回 (记录数, 文档数)。"""
    raws = session.scalars(
        select(RawEvidence).where(
            RawEvidence.consolidated == 0, RawEvidence.deleted_at.is_(None)
        )
    ).all()
    for raw in raws:
        queue.submit(consolidate_record, settings, raw.id, lane="record")

    docs = session.scalars(
        select(Document).where(Document.status == "processing", Document.deleted_at.is_(None))
    ).all()
    requeued_docs = 0
    for doc in docs:
        if not doc.file_path:
            continue
        queue.submit(
            process_document, settings, doc.id, doc.file_path, "auto", doc.allow_cloud,
            doc.multimodal, lane="doc",
        )
        requeued_docs += 1

    return len(raws), requeued_docs
