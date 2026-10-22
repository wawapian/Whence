"""时间显示：UTC 存储 → 本机时区带偏移输出（用户实测：下午一点记录显示凌晨五点）。"""
import re
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app import deps
from app.api.export import export_md
from app.config import Settings
from app.main import create_app
from app.memory import documents, store
from app.models.db import init_db
from app.models.schemas import ExtractedFact


def test_to_local_converts_naive_utc():
    tz8 = timezone(timedelta(hours=8))
    # naive-UTC 05:00 → UTC+8 即北京 13:00
    assert store.to_local(datetime(2026, 10, 14, 5, 0), tz=tz8).hour == 13
    # 已带时区：按同一时刻转换，不重复解释
    aware = datetime(2026, 10, 14, 5, 0, tzinfo=timezone.utc)
    assert store.to_local(aware, tz=tz8).hour == 13
    assert store.to_local(None) is None


def test_documents_api_returns_offset_timestamps(tmp_path):
    """接口时间带时区偏移——前端直接切片显示即是本机时间（13 点记录不再显示 5 点）。"""
    app = create_app()
    app.dependency_overrides[deps.get_settings] = lambda: Settings(
        data_dir=tmp_path, embedding_model="BAAI/bge-small-zh-v1.5"
    )
    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    init_db(engine, dim=512)  # 与 override 的模型一致（维度护栏会校验）
    with Session(engine) as s:
        documents.insert_document(s, kind="word", title="a.docx", status="done")
        s.commit()

    listed = TestClient(app).get("/api/documents").json()["documents"]
    created = listed[0]["created_at"]
    assert re.search(r"[+-]\d{2}:\d{2}$", created), f"缺少时区偏移: {created}"


def test_export_uses_local_time(tmp_path):
    """导出 Markdown 的时间行按本机时区渲染（与接口同一次转换）。"""
    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    init_db(engine)
    with Session(engine) as s:
        raw = store.insert_raw(s, kind="text", content="内容")
        s.flush()
        fact = store.insert_fact(
            s, ExtractedFact(type="other", summary="摘要", confidence=0.9), raw.id
        )
        s.commit()
        resp = export_md(session=s)

    body = resp.body.decode("utf-8")
    expected = store.to_local(fact.created_at).strftime("%Y-%m-%d %H:%M")
    assert f"## {expected} · other · 摘要" in body
