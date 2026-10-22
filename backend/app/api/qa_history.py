"""提问历史接口：翻查（GET）与全清（DELETE）；写入在问答路径内落行，无单独写端点。"""
from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select, or_, text
from datetime import datetime, timedelta, timezone
from typing import Literal
from sqlalchemy.orm import Session

from app.deps import get_session
from app.memory import qa_history
from app.memory.store import to_local
from app.models.db import QaHistory

router = APIRouter(prefix="/api")


@router.get("/qa-history")
def get_qa_history(
    limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0),
    q: str = "", period: Literal["all", "week", "month", "earlier"] = "all",
    session: Session = Depends(get_session)
):
    """提问历史列表（created_at 倒序）；`total` = 全表条数，供翻页端用。"""
    conditions = []
    if q.strip():
        conditions.append(or_(QaHistory.question.contains(q.strip(), autoescape=True),
                              QaHistory.answer.contains(q.strip(), autoescape=True)))
    local = datetime.now().astimezone()
    if period == "week":
        boundary = (local.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=6)).astimezone(timezone.utc)
        conditions.append(QaHistory.created_at >= boundary)
    elif period in {"month", "earlier"}:
        boundary = local.replace(day=1, hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)
        conditions.append(QaHistory.created_at >= boundary if period == "month" else QaHistory.created_at < boundary)
    rows = session.scalars(select(QaHistory).where(*conditions).order_by(
        QaHistory.created_at.desc(), text('rowid DESC')).limit(limit).offset(offset)).all()
    total = session.scalar(select(func.count()).select_from(QaHistory).where(*conditions)) or 0
    return {
        "items": [
            {
                "id": r.id,
                # ORM 读回是 naive-UTC：经 to_local 出口（带本机偏移），勿手拼 isoformat
                "created_at": to_local(r.created_at),
                "question": r.question,
                "resolved_question": r.resolved_question,
                "answer": r.answer,
                "from_cache": bool(r.from_cache),
            }
            for r in rows
        ],
        "total": total,
    }


@router.delete("/qa-history", status_code=204)
def clear_qa_history(session: Session = Depends(get_session)):
    """全清提问历史（不可恢复）；空表同样 204。"""
    qa_history.clear_all(session)
    session.commit()
