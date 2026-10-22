"""Markdown 导出接口。"""
from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.deps import get_session
from app.memory.store import to_local
from app.models.db import Fact, RawEvidence

router = APIRouter(prefix="/api")


@router.get("/export")
def export_md(session: Session = Depends(get_session)):
    facts = session.scalars(
        select(Fact)
        .join(RawEvidence, RawEvidence.id == Fact.raw_id)
        .where(RawEvidence.deleted_at.is_(None))
        .order_by(Fact.created_at)
    ).all()
    lines = ["# 记忆导出", ""]
    for f in facts:
        raw = session.get(RawEvidence, f.raw_id) if f.raw_id else None
        local_dt = to_local(f.created_at)
        lines.append(f"## {local_dt:%Y-%m-%d %H:%M} · {f.type} · {f.summary or ''}")
        if raw and raw.content:
            lines.append(f"- 原文：{raw.content}")
        if raw and raw.file_path:
            lines.append(f"- 文件：![]({raw.file_path})")
        lines.append("")
    return PlainTextResponse("\n".join(lines), media_type="text/markdown")
