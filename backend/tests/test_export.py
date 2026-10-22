from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app import deps
from app.config import Settings
from app.main import create_app
from app.memory import store
from app.models.db import init_db
from app.models.schemas import ExtractedFact


def test_export_contains_fact(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    init_db(engine, dim=1024)
    session = Session(engine)
    raw = store.insert_raw(session, kind="text", content="身份证放在书房抽屉")
    session.flush()
    store.insert_fact(
        session,
        ExtractedFact(type="item_location", summary="身份证放书房", confidence=0.9),
        raw.id,
    )
    session.commit()

    app = create_app()
    app.dependency_overrides[deps.get_settings] = lambda: Settings(data_dir=tmp_path)
    client = TestClient(app)
    resp = client.get("/api/export")
    assert resp.status_code == 200
    assert "身份证" in resp.text
    assert "item_location" in resp.text
