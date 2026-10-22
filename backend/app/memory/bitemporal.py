"""双时态 upsert：状态类记忆新值覆盖、旧值 SUPERSEDED 留档。"""
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.db import Fact
from app.models.schemas import STATE_TYPES, ExtractedFact
from app.memory import store


def upsert(session: Session, fact: ExtractedFact, raw_id: str, user_id: str = "me", state: str = "confirmed") -> Fact:
    """状态类且带 subject_key 时，关闭旧值、插入新值；否则直接插入。"""
    if fact.type in STATE_TYPES and fact.subject_key:
        current = session.scalar(
            select(Fact)
            .where(
                Fact.user_id == user_id,
                Fact.subject_key == fact.subject_key,
                Fact.valid_to.is_(None),
                Fact.archived == 0,
            )
            .order_by(Fact.valid_from.desc())
        )
        new = store.insert_fact(session, fact, raw_id, state=state)
        if current is not None:
            session.flush()  # 新 fact 先落库，满足旧值 superseded_by 外键
            current.valid_to = store.utcnow()
            current.superseded_by = new.id
        return new
    return store.insert_fact(session, fact, raw_id, state=state)
