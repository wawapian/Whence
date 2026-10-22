from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.memory import bitemporal, store
from app.models.db import Fact, init_db
from app.models.schemas import ExtractedFact


def _session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    init_db(engine)
    return Session(engine)


def _raw_id(session, content="x"):
    raw = store.insert_raw(session, kind="text", content=content)
    session.flush()
    return raw.id


def _fact(subject_key, summary, type="item_location"):
    return ExtractedFact(type=type, subject_key=subject_key, summary=summary, confidence=0.9)


def test_state_supersedes_old(tmp_path):
    session = _session(tmp_path)
    f1 = bitemporal.upsert(session, _fact("身份证:位置", "放书房抽屉"), _raw_id(session))
    f2 = bitemporal.upsert(session, _fact("身份证:位置", "放背包夹层"), _raw_id(session))
    session.commit()

    f1_db = session.get(Fact, f1.id)
    f2_db = session.get(Fact, f2.id)
    assert f1_db.valid_to is not None       # 旧值被关闭
    assert f1_db.superseded_by == f2.id     # 指向新值
    assert session.query(Fact).count() == 2  # 旧值未删除，留档


def test_event_does_not_supersede(tmp_path):
    session = _session(tmp_path)
    f1 = bitemporal.upsert(session, _fact(None, "买了鸡蛋", type="consumption"), _raw_id(session))
    f2 = bitemporal.upsert(session, _fact(None, "买了牛奶", type="consumption"), _raw_id(session))
    session.commit()

    assert session.query(Fact).filter(Fact.valid_to.is_(None)).count() == 2  # 事件只增不减
