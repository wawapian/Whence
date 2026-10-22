from datetime import datetime, timezone

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app import deps
from app.config import Settings
from app.core import answer, extraction, llm, queue
from app.main import create_app
from app.memory import fts, vector
from app.models.db import Fact, RawEvidence, init_db


def _boom(*a, **k):
    raise RuntimeError("boom")


def test_extract_failure_records_pending(tmp_path, monkeypatch):
    app = create_app()
    app.dependency_overrides[deps.get_settings] = lambda: Settings(
        data_dir=tmp_path, embedding_model="BAAI/bge-small-zh-v1.5"
    )
    monkeypatch.setattr(extraction, "extract", _boom)
    monkeypatch.setattr(llm, "embed", lambda text, model=None: [0.1] * 512)

    def _submit(fn, *args, **kwargs):  # 记录整理内联执行：否则后台线程与下面断言竞态
        lane = kwargs.pop("lane", "doc")  # 车道参数由队列消费，不进任务函数
        if lane == "record":
            return fn(*args, **kwargs) or "test-inline"
        return "test-noop"

    monkeypatch.setattr(queue, "submit", _submit)

    client = TestClient(app)
    resp = client.post("/api/memories", data={"text": "今天买了鸡蛋"})
    assert resp.status_code == 200
    assert resp.json()["state"] == "accepted"  # accept-fast；整理（内联）后事实回落 pending

    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    with Session(engine) as s:
        assert s.query(RawEvidence).count() == 1  # 原文不丢
        assert s.query(Fact).filter(Fact.state == "pending").count() == 1


def test_answer_without_llm_returns_raw_list(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    init_db(engine)
    session = Session(engine)
    session.add(
        Fact(id="f1", type="item_location", summary="身份证放在书房抽屉",
             valid_from=datetime.now(timezone.utc))
    )
    session.flush()
    fts.index_fact(session, "f1", "身份证放在书房抽屉")
    vector.index_fact(session, "f1", [0.1] * 512)
    session.commit()

    monkeypatch.setattr(llm, "embed", lambda q, model=None: [0.1] * 512)
    monkeypatch.setattr(answer, "synthesize", _boom)

    resp = answer.answer(session, "身份证在哪", Settings(data_dir=tmp_path, qa_engine="legacy"))
    assert resp.sources  # 仍返回候选原文
    assert "失败" in resp.answer
