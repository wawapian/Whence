"""提醒接口。"""
import re
import logging
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, model_validator
from sqlalchemy import or_, select, text
from sqlalchemy.orm import Session

from app.deps import get_session, get_settings, _session_factory
from app.models.db import Fact, RawEvidence, Material, VecMap
from app.memory import qa_history
from app.memory.store import to_local
from app.memory import fts, store, materials, vector
from app.core import llm, queue
from app.config import Settings
from app.models.schemas import ExtractedFact

router = APIRouter(prefix="/api")
logger = logging.getLogger(__name__)


def _parse_datetime_hint(hint: str | None, reference: datetime | None = None) -> datetime | None:
    """Anchor a supported time expression to its original record time (aware datetime)."""
    if not hint:
        return None
    hint = hint.strip()
    now = to_local(reference or datetime.now(timezone.utc))
    try:
        explicit = datetime.fromisoformat(hint)
        if explicit.tzinfo is not None:
            return explicit
        return explicit.replace(tzinfo=now.tzinfo, hour=9 if len(hint) == 10 else explicit.hour)
    except ValueError:
        pass
    # Common spoken Mandarin hours: 三点/两点/十一点/下午三点半.
    def number(value):
        if value.isdigit():
            return int(value)
        digits = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
                  "六": 6, "七": 7, "八": 8, "九": 9}
        if "十" in value:
            left, right = value.split("十", 1)
            return digits.get(left, 1) * 10 + digits.get(right, 0)
        return digits.get(value, -1)

    clock = re.search(r"([一二三四五六七八九十两\d]{1,3})[点时:：](\d{1,2})?", hint)
    hour = number(clock.group(1)) if clock else 19 if "晚上" in hint else 15 if "下午" in hint else 12 if "中午" in hint else 9
    minute = int(clock.group(2) or 0) if clock else 0
    if clock and "半" in hint[clock.end():]:
        minute = 30
    if clock and ("下午" in hint or "晚上" in hint) and hour < 12:
        hour += 12
    offset = 1 if "明天" in hint else 2 if "后天" in hint else 0 if "今天" in hint else None
    weekday = re.search(r"(下周|本周|这周|周|星期)([一二三四五六日天])", hint)
    if offset is None and weekday:
        weekday_number = "一二三四五六日".index(weekday.group(2).replace("天", "日"))
        offset = weekday_number - now.weekday()
        if weekday.group(1) == "下周":
            offset += 7
        elif offset < 0 and weekday.group(1) in {"周", "星期"}:
            offset += 7
    if offset is not None:
        target = now + timedelta(days=offset)
        try:
            return target.replace(hour=hour, minute=minute, second=0, microsecond=0)
        except ValueError:
            return None
    m = re.search(r"(\d{1,2})月(\d{1,2})日", hint)
    if m:
        month, day = int(m.group(1)), int(m.group(2))
        try:
            dt = now.replace(month=month, day=day, hour=hour, minute=minute, second=0, microsecond=0)
            if dt < now:
                dt = dt.replace(year=now.year + 1)
            return dt
        except ValueError:
            return None
    return None


@router.get("/reminders")
def reminders(session: Session = Depends(get_session)):
    facts = session.scalars(
        select(Fact)
        .outerjoin(RawEvidence, RawEvidence.id == Fact.raw_id)
        .where(
            Fact.valid_to.is_(None),
            Fact.archived == 0,
            # 只排除已进回收站的记录；无 raw 的孤儿 fact 照旧可见（过滤不改其它语义）
            or_(RawEvidence.id.is_(None), RawEvidence.deleted_at.is_(None)),
        )
    ).all()
    due, pending, completed = [], [], []
    for f in facts:
        fields = f.fields or {}
        if not fields.get("is_reminder"):
            continue
        hint = fields.get("datetime_hint")
        target = _parse_datetime_hint(fields.get("reminder_due_at") or hint, f.created_at)
        item = {"fact_id": f.id, "raw_id": f.raw_id, "summary": f.summary, "datetime_hint": hint,
                "due_at": to_local(target), "completed_at": fields.get("reminder_completed_at")}
        if fields.get("reminder_completed"):
            completed.append(item)
        elif target is not None:
            due.append(item)
        else:
            pending.append(item)
    due.sort(key=lambda item: item["due_at"])
    return {"due": due, "pending": pending, "completed": completed}


class ReminderUpdate(BaseModel):
    completed: bool | None = None
    due_at: datetime | None = None

    @model_validator(mode="after")
    def validate_update(self):
        if not self.model_fields_set:
            raise ValueError("请选择完成状态或提醒时间")
        if self.due_at is not None and self.due_at.utcoffset() is None:
            raise ValueError("提醒时间必须包含时区")
        return self


@router.patch("/reminders/{fact_id}")
def update_reminder(fact_id: str, payload: ReminderUpdate, session: Session = Depends(get_session)):
    fact = session.get(Fact, fact_id)
    raw = session.get(RawEvidence, fact.raw_id) if fact and fact.raw_id else None
    if (fact is None or fact.valid_to is not None or fact.archived or
        not (fact.fields or {}).get("is_reminder") or (raw and raw.deleted_at is not None)):
        raise HTTPException(404, "提醒不存在或已归档")
    fields = dict(fact.fields or {})
    if payload.completed is not None:
        fields["reminder_completed"] = payload.completed
        fields["reminder_completed_at"] = datetime.now(timezone.utc).isoformat() if payload.completed else None
    if "due_at" in payload.model_fields_set:
        fields["reminder_due_at"] = payload.due_at.isoformat() if payload.due_at else None
    fact.fields = fields
    qa_history.bump(session)
    session.commit()
    return {"ok": True}


class ReminderCreate(BaseModel):
    summary: str = Field(min_length=1, max_length=1000)
    due_at: datetime | None = None

    @model_validator(mode="after")
    def validate_create(self):
        self.summary = self.summary.strip()
        if not self.summary:
            raise ValueError("提醒内容不能为空")
        if self.due_at is not None and self.due_at.utcoffset() is None:
            raise ValueError("提醒时间必须包含时区")
        return self


@router.post("/reminders", status_code=201)
def create_reminder(payload: ReminderCreate, session: Session = Depends(get_session),
                    settings: Settings = Depends(get_settings)):
    """Explicit user-entered facts need no LLM extraction; use the same RAW/FACT/material spine."""
    raw = store.insert_raw(session, kind="text", content=payload.summary)
    raw.consolidated = 1
    session.flush()
    fact = store.insert_fact(session, ExtractedFact(type="todo", summary=payload.summary,
                             is_reminder=True, confidence=1), raw.id)
    fact.fields = {**fact.fields, "reminder_due_at": payload.due_at.isoformat() if payload.due_at else None}
    session.flush()
    fts.index_fact(session, fact.id, payload.summary)
    material = Material(id=store.new_id(), source_type="record", source_id=raw.id,
                        title=payload.summary[:40], summary=payload.summary[:200])
    session.add(material)
    session.flush()
    session.execute(text("INSERT INTO material_fts(content, material_id) VALUES (:c, :id)"),
                    {"c": payload.summary, "id": material.id})
    session.commit()
    try:
        queue.submit(_index_created_reminder, settings, raw.id, fact.id, lane="record")
    except Exception:
        logger.warning("提醒语义索引未能入队；原文、提醒与关键词索引已保存", exc_info=True)
    return {"raw_id": raw.id, "fact_id": fact.id}


def _index_created_reminder(settings: Settings, raw_id: str, fact_id: str):
    """Attach local vectors to an explicitly entered fact using the existing index functions."""
    with _session_factory(settings)() as session:
        raw = session.get(RawEvidence, raw_id)
        fact = session.get(Fact, fact_id)
        if raw is None or raw.deleted_at is not None or fact is None:
            return
        try:
            if not session.scalar(select(VecMap.fact_id).where(VecMap.fact_id == fact_id)):
                vec = llm.embed(raw.content or "", model=settings.embedding_model)
                vector.index_fact(session, fact_id, vec)
            materials.upsert_for_record(session, raw, title=fact.summary, model=settings.embedding_model)
            qa_history.bump(session)
            session.commit()
        except Exception:
            session.rollback()
            logger.warning("提醒语义索引暂不可用；原文、提醒与关键词索引仍可用", exc_info=True)
