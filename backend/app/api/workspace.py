"""Workspace capabilities; reuse the existing evidence, ingest and trash spine."""
import json
import re
from io import BytesIO
from pathlib import PurePosixPath
from zipfile import ZipFile, ZIP_DEFLATED

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from app.api.files import process_document, readable_original_path
from app.config import Settings
from app.core import queue, trash
from app.deps import get_session, get_settings
from app.memory import qa_history
from app.memory.store import to_local
from app.models.db import Document, Fact, QaHistory, RawEvidence

router = APIRouter(prefix="/api/workspace", tags=["workspace"])


@router.get("/spending")
def spending(period: str = "this_month", session: Session = Depends(get_session)):
    from app.core.memory_stats import collect
    try:
        result = collect(session, period=period, start=None, end=None, kind="spending", type_filter="consumption")
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    facts = {f.id: f.raw_id for f in session.scalars(select(Fact).where(Fact.id.in_([r.fact_id for r in result.rows])))}
    return {"start": result.start, "end": result.end, "totals": result.totals,
            "count": result.count, "missing": result.missing,
            "rows": [{"fact_id": r.fact_id, "raw_id": facts.get(r.fact_id), "amount": r.amount,
                      "currency": r.currency, "summary": r.summary, "occurred": to_local(r.occurred)} for r in result.rows]}


@router.get("/overview")
def overview(session: Session = Depends(get_session), settings: Settings = Depends(get_settings)):
    docs = session.scalars(select(Document).where(Document.deleted_at.is_(None))).all()
    total_bytes = 0
    for doc in docs:
        path = readable_original_path(session, settings.data_dir, doc=doc)
        try:
            total_bytes += path.stat().st_size if path else 0
        except OSError:
            pass
    count = lambda model, condition: session.scalar(select(func.count()).select_from(model).where(condition)) or 0
    return {
        "records": count(RawEvidence, RawEvidence.deleted_at.is_(None)),
        "documents": len(docs), "storage_bytes": total_bytes,
        "history": session.scalar(select(func.count()).select_from(QaHistory)) or 0,
        "trash": count(RawEvidence, RawEvidence.deleted_at.is_not(None)) + count(Document, Document.deleted_at.is_not(None)),
        "processing": sum(d.status == "processing" for d in docs),
        "cloud_configured": bool(settings.llm_api_key),
        "attachment_cloud_opt_in": True,
        "storage": "SQLite + 本地原件", "trash_retention_days": trash.TRASH_TTL_DAYS,
    }


def _fact_json(f):
    return {"fact_id": f.id, "type": f.type, "summary": f.summary, "fields": f.fields or {},
            "subject_key": f.subject_key, "valid_from": to_local(f.valid_from),
            "valid_to": to_local(f.valid_to), "created_at": to_local(f.created_at),
            "raw_id": f.raw_id, "superseded_by": f.superseded_by}


@router.get("/records/{raw_id}")
def record_detail(raw_id: str, session: Session = Depends(get_session)):
    raw = session.get(RawEvidence, raw_id)
    if raw is None or raw.deleted_at is not None:
        raise HTTPException(404, "记录不存在或已移入回收站")
    facts = session.scalars(select(Fact).where(Fact.raw_id == raw_id).order_by(Fact.created_at)).all()
    subjects = {f.subject_key for f in facts if f.subject_key}
    history = session.scalars(select(Fact).join(RawEvidence, RawEvidence.id == Fact.raw_id).where(
        Fact.subject_key.in_(subjects), RawEvidence.deleted_at.is_(None)
    ).order_by(Fact.valid_from.desc())).all() if subjects else []
    docs = session.scalars(select(Document).where(Document.raw_id == raw_id, Document.deleted_at.is_(None))).all()
    return {"id": raw.id, "content": raw.content, "created_at": to_local(raw.created_at),
            "consolidated": raw.consolidated, "facts": [_fact_json(f) for f in facts],
            "history": [_fact_json(f) for f in history],
            "documents": [{"id": d.id, "title": d.title, "status": d.status} for d in docs]}


@router.get("/history/{history_id}")
def history_detail(history_id: str, session: Session = Depends(get_session)):
    row = session.get(QaHistory, history_id)
    if row is None:
        raise HTTPException(404, "提问历史不存在")
    # Historical answers remain verbatim; raw endpoints still enforce current visibility.
    execution = json.loads(row.execution or "{}")
    return {"id": row.id, "question": row.question, "answer": row.answer,
            "execution": execution, "status": execution.get("status", "ok"),
            "resolved_question": row.resolved_question, "from_cache": bool(row.from_cache),
            "created_at": to_local(row.created_at), "sources": json.loads(row.sources),
            "documents": json.loads(row.documents)}


class ExportSelection(BaseModel):
    document_ids: list[str] = Field(min_length=1, max_length=100)


@router.post("/files/export")
def export_files(payload: ExportSelection, session: Session = Depends(get_session),
                 settings: Settings = Depends(get_settings)):
    selected = []
    for doc_id in dict.fromkeys(payload.document_ids):
        doc = session.get(Document, doc_id)
        path = readable_original_path(session, settings.data_dir, doc=doc)
        if path is None:
            raise HTTPException(404, "所选原件不可用，请刷新后重试")
        selected.append((doc, path))
    if sum(path.stat().st_size for _, path in selected) > 256 * 1024 * 1024:
        raise HTTPException(413, "本次导出超过 256 MB，请分批下载")
    buffer = BytesIO()
    names = set()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        for doc, path in selected:
            name = PurePosixPath((doc.title or path.name).replace("\\", "/")).name
            name = re.sub(r'[\x00-\x1f<>:"|?*]', "_", name).strip(". ") or "file"
            original = name
            n = 1
            while name.casefold() in names:
                name = f"{n}_{original}"
                n += 1
            names.add(name.casefold())
            archive.write(path, name)
    return Response(buffer.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": 'attachment; filename="memory-originals.zip"'})


@router.post("/files/{doc_id}/retry")
def retry_file(doc_id: str, session: Session = Depends(get_session), settings: Settings = Depends(get_settings)):
    doc = session.get(Document, doc_id)
    if readable_original_path(session, settings.data_dir, doc=doc) is None:
        raise HTTPException(404, "原件不可用，请重新上传")
    # A failed ingestion transaction rolls back chunks; reject partial results to avoid duplicates.
    from app.models.db import Chunk, Image, DocTable
    if any(session.scalar(select(func.count()).select_from(model).where(model.document_id == doc_id))
           for model in (Chunk, Image, DocTable)):
        raise HTTPException(409, "该文件已有解析结果，请重新上传以重新解析")
    claimed = session.execute(update(Document).where(Document.id == doc_id, Document.status == "failed",
                              Document.deleted_at.is_(None)).values(status="processing")).rowcount
    if not claimed:
        session.rollback()
        raise HTTPException(409, "仅解析失败的文件可以重试")
    session.commit()
    try:
        queue.submit(process_document, settings, doc.id, doc.file_path, "auto", doc.allow_cloud, doc.multimodal)
    except Exception:
        doc.status = "failed"
        session.commit()
        raise HTTPException(503, "解析队列暂不可用，请稍后重试")
    return {"status": "processing", "document_id": doc.id}


@router.post("/trash/{kind}/{item_id}/restore-group")
def restore_group(kind: str, item_id: str, session: Session = Depends(get_session)):
    if kind not in {"records", "documents"}:
        raise HTTPException(404, "资源类型不存在")
    item = session.get(RawEvidence if kind == "records" else Document, item_id)
    if item is None or item.deleted_at is None:
        raise HTTPException(404, "该条目不在回收站")
    raw = item if kind == "records" else trash._group_record(session, item)
    docs = trash._group_documents(session, raw, deleted=True) if raw else [item]
    count = 0
    for doc in docs:
        count += int(trash.restore_document(session, doc.id))
    if raw:
        count += int(trash.restore_record(session, raw.id))
    session.commit()
    return {"restored": count}
