"""回收站接口：列表 / 恢复 / 彻底删除（软删入口在各自资源的 api 模块）。"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core import trash
from app.deps import get_session
from app.models.db import Document, RawEvidence

router = APIRouter(prefix="/api")


@router.get("/trash")
def get_trash(session: Session = Depends(get_session)):
    """回收站内容：记录 + 文档（含 `deleted_at` / `days_left`，由 `trash.list_trash` 产出）。"""
    return trash.list_trash(session)


@router.delete("/trash")
def empty_trash(session: Session = Depends(get_session)):
    """清空回收站：站内全部对象彻底删除（逐条、单条失败跳过），返回 `{records, documents}` 成功计数。

    不针对具体 id，故空站返回 0 计数而非 404；活对象（未软删）不在动作范围内。
    """
    return trash.empty_trash(session)


@router.post("/trash/records/{raw_id}/restore")
def restore_trash_record(raw_id: str, session: Session = Depends(get_session)):
    """恢复记录：抹 `deleted_at`；不在回收站或不存在 → 404。"""
    if not trash.restore_record(session, raw_id):
        raise HTTPException(status_code=404, detail="record not in trash")
    session.commit()
    return {"ok": True}


@router.post("/trash/documents/{doc_id}/restore")
def restore_trash_document(doc_id: str, session: Session = Depends(get_session)):
    """恢复文档：抹 `deleted_at`；不在回收站或不存在 → 404。"""
    if not trash.restore_document(session, doc_id):
        raise HTTPException(status_code=404, detail="document not in trash")
    session.commit()
    return {"ok": True}


@router.delete("/trash/records/{raw_id}", status_code=204)
def purge_trash_record(raw_id: str, session: Session = Depends(get_session)):
    """彻底删除回收站内的记录（含事实链、索引与附件文件）；不在回收站或不存在 → 404。

    `/api/trash/*` 只对回收站内的对象动作：未软删的活对象一律 404，不做隐式动作。
    失败直接上抛由 FastAPI 处理。
    """
    raw = session.get(RawEvidence, raw_id)
    if raw is None or raw.deleted_at is None:
        raise HTTPException(status_code=404, detail="record not in trash")
    # 守卫已排除 None/活对象，此处返回恒 True，故不检查布尔返回
    trash.purge_record(session, raw_id)
    session.commit()


@router.delete("/trash/documents/{doc_id}", status_code=204)
def purge_trash_document(doc_id: str, session: Session = Depends(get_session)):
    """彻底删除回收站内的文档（含分块、索引、表格、图片与磁盘文件）；不在回收站或不存在 → 404。"""
    doc = session.get(Document, doc_id)
    if doc is None or doc.deleted_at is None:
        raise HTTPException(status_code=404, detail="document not in trash")
    # 守卫已排除 None/活对象，此处返回恒 True，故不检查布尔返回
    trash.purge_document(session, doc_id)
    session.commit()
