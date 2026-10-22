from datetime import datetime, timezone

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app import deps
from app.config import Settings
from app.main import create_app
from app.models.db import Fact, init_db


def test_reminder_due_and_pending(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    init_db(engine, dim=1024)
    session = Session(engine)
    now = datetime.now(timezone.utc)
    session.add(Fact(id="r1", type="todo", summary="交报告",
                     fields={"is_reminder": True, "datetime_hint": "明天"},
                     valid_from=now))
    session.add(Fact(id="r2", type="todo", summary="打折",
                     fields={"is_reminder": True, "datetime_hint": None},
                     valid_from=now))
    session.add(Fact(id="r3", type="consumption", summary="买鸡蛋",
                     fields={"is_reminder": False},
                     valid_from=now))
    session.commit()

    app = create_app()
    app.dependency_overrides[deps.get_settings] = lambda: Settings(data_dir=tmp_path)
    client = TestClient(app)
    resp = client.get("/api/reminders")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["due"]) == 1
    assert len(body["pending"]) == 1
    assert body["due"][0]["fact_id"] == "r1"
    assert body["pending"][0]["fact_id"] == "r2"
