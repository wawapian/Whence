"""Workspace integration: real isolated SQLite, files, visibility and persisted controls."""
from datetime import datetime, timedelta, timezone
from io import BytesIO
from zipfile import ZipFile

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app import deps
from app.config import Settings
from app.core import queue
from app.main import create_app
from app.memory import qa_history, store
from app.models.db import Document, Fact, RawEvidence, Chunk


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    settings = Settings(_env_file=None, data_dir=tmp_path, embedding_model="BAAI/bge-small-zh-v1.5", warmup=False)
    app = create_app()
    app.dependency_overrides[deps.get_settings] = lambda: settings
    monkeypatch.setattr(queue, "submit", lambda *args, **kw: "test-queued")
    client = TestClient(app)  # No lifespan: production store never opened.
    yield client, deps._session_factory(settings), tmp_path


def add_file(factory, root, title="原件.txt", deleted=False, path="files/sample.txt", raw_id=None, status="done"):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("本地原件正文", encoding="utf-8")
    with factory() as s:
        doc = Document(id=store.new_id(), title=title, kind="document", status=status, file_path=path,
                       raw_id=raw_id, allow_cloud=0, deleted_at=datetime.now(timezone.utc) if deleted else None)
        s.add(doc); s.commit()
        return doc.id


def test_reminder_create_complete_reopen_and_fixed_time(workspace):
    c, factory, root = workspace
    value = "2026-10-20T09:00:00+08:00"
    r = c.post("/api/reminders", json={"summary": "交房租", "due_at": value})
    assert r.status_code == 201
    fact_id, raw_id = r.json()["fact_id"], r.json()["raw_id"]
    due = c.get("/api/reminders").json()["due"][0]
    assert datetime.fromisoformat(due["due_at"]) == datetime.fromisoformat(value)
    assert c.patch(f"/api/reminders/{fact_id}", json={"completed": True}).status_code == 200
    assert c.get("/api/reminders").json()["due"] == []
    assert c.get("/api/reminders").json()["completed"][0]["completed_at"]
    assert c.patch(f"/api/reminders/{fact_id}", json={"completed": False}).status_code == 200
    assert len(c.get("/api/reminders").json()["due"]) == 1
    assert c.get(f"/api/workspace/records/{raw_id}").json()["content"] == "交房租"
    assert c.delete(f"/api/memories/{raw_id}").status_code == 204
    assert c.get("/api/reminders").json() == {"due": [], "pending": [], "completed": []}
    assert c.patch(f"/api/reminders/{fact_id}", json={"completed": True}).status_code == 404


def test_reminder_validation_and_pending(workspace):
    c, *_ = workspace
    assert c.post("/api/reminders", json={"summary": "   "}).status_code == 422
    assert c.post("/api/reminders", json={"summary": "test", "due_at": "2026-10-20T09:00"}).status_code == 422
    r = c.post("/api/reminders", json={"summary": "找时间交报告"}).json()
    assert c.get("/api/reminders").json()["pending"][0]["fact_id"] == r["fact_id"]
    assert c.patch(f'/api/reminders/{r["fact_id"]}', json={}).status_code == 422
    assert c.patch(f'/api/reminders/{r["fact_id"]}', json={"due_at": "2026-10-20T09:00:00+08:00"}).status_code == 200
    assert c.get("/api/reminders").json()["pending"] == []


def test_relative_reminder_time_is_anchored_to_creation():
    from app.api.reminders import _parse_datetime_hint
    reference = datetime(2025, 10, 1, 1, tzinfo=timezone.utc)
    value = _parse_datetime_hint("明天下午3点", reference)
    assert value.date() == store.to_local(reference).date() + timedelta(days=1)
    assert value.hour == 15
    assert _parse_datetime_hint("明天下午三点半", reference).hour == 15
    assert _parse_datetime_hint("明天下午三点半", reference).minute == 30
    assert _parse_datetime_hint("下周一上午十点", reference).weekday() == 0
    assert _parse_datetime_hint("下周一上午十点", reference).hour == 10
    assert _parse_datetime_hint("2026-10-20", reference).day == 20
    assert _parse_datetime_hint("13月99日", reference) is None


def test_overview_and_document_metadata_are_real(workspace):
    c, factory, root = workspace
    doc_id = add_file(factory, root)
    add_file(factory, root, deleted=True, path="files/deleted.txt")
    overview = c.get("/api/workspace/overview").json()
    assert overview["documents"] == 1
    assert overview["storage_bytes"] == len("本地原件正文".encode())
    doc = c.get("/api/documents").json()["documents"][0]
    assert doc["size_bytes"] == overview["storage_bytes"]
    assert doc["id"] == doc_id and doc["allow_cloud"] is False


def test_zip_retains_unicode_names_and_disambiguates_duplicates(workspace):
    c, factory, root = workspace
    a = add_file(factory, root, "凭证.txt", path="files/a.txt")
    b = add_file(factory, root, "凭证.txt", path="files/b.txt")
    r = c.post("/api/workspace/files/export", json={"document_ids": [a, b, a]})
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    with ZipFile(BytesIO(r.content)) as archive:
        assert archive.namelist() == ["凭证.txt", "1_凭证.txt"]
        assert archive.read("凭证.txt").decode() == "本地原件正文"


def test_zip_rejects_deleted_missing_and_escaping_paths(workspace):
    c, factory, root = workspace
    doc_id = add_file(factory, root, deleted=True)
    assert c.post("/api/workspace/files/export", json={"document_ids": [doc_id]}).status_code == 404
    assert c.post("/api/workspace/files/export", json={"document_ids": ["absent"]}).status_code == 404
    with factory() as s:
        doc = s.get(Document, doc_id); doc.deleted_at = None; doc.file_path = "../outside.txt"; s.commit()
    assert c.post("/api/workspace/files/export", json={"document_ids": [doc_id]}).status_code == 404
    assert c.post("/api/workspace/files/export", json={"document_ids": []}).status_code == 422


def test_zip_normalizes_untrusted_upload_name(workspace):
    c, factory, root = workspace
    doc_id = add_file(factory, root, "../../outside.txt")
    response = c.post("/api/workspace/files/export", json={"document_ids": [doc_id]})
    with ZipFile(BytesIO(response.content)) as archive:
        assert archive.namelist() == ["outside.txt"]


def test_retry_only_failed_visible_document_and_no_duplicate_jobs(workspace, monkeypatch):
    c, factory, root = workspace
    submitted = []
    monkeypatch.setattr(queue, "submit", lambda *args, **kw: submitted.append(args))
    doc_id = add_file(factory, root, status="failed")
    assert c.post(f"/api/workspace/files/{doc_id}/retry").status_code == 200
    assert c.post(f"/api/workspace/files/{doc_id}/retry").status_code == 409
    assert len(submitted) == 1
    assert c.get(f"/api/files/{doc_id}").json()["status"] == "processing"
    c.delete(f"/api/documents/{doc_id}")
    assert c.post(f"/api/workspace/files/{doc_id}/retry").status_code == 404


def test_retry_refuses_partial_existing_chunks(workspace):
    c, factory, root = workspace
    doc_id = add_file(factory, root, status="failed")
    with factory() as s:
        s.add(Chunk(id="existing", document_id=doc_id, seq=0, text="保留的结果", parent_text="保留的结果")); s.commit()
    assert c.post(f"/api/workspace/files/{doc_id}/retry").status_code == 409
    assert c.get(f"/api/files/{doc_id}").json()["chunk_count"] == 1


def test_group_restore_is_atomic_and_old_individual_route_survives(workspace):
    c, factory, root = workspace
    with factory() as s:
        raw = store.insert_raw(s, "document", content="整组记录"); s.commit(); raw_id = raw.id
    a = add_file(factory, root, raw_id=raw_id, path="files/a.txt")
    b = add_file(factory, root, raw_id=raw_id, path="files/b.txt")
    c.delete(f"/api/documents/{a}")
    assert c.get(f"/api/files/{b}/raw").status_code == 404
    assert c.post(f"/api/workspace/trash/documents/{a}/restore-group").json()["restored"] == 3
    assert c.get("/api/trash").json() == {"records": [], "documents": []}
    assert c.get(f"/api/files/{a}/raw").status_code == 200
    c.delete(f"/api/memories/{raw_id}")
    assert c.post(f"/api/trash/records/{raw_id}/restore").status_code == 200
    assert len(c.get("/api/trash").json()["documents"]) == 2


def test_history_search_pagination_and_full_sources(workspace):
    c, factory, root = workspace
    with factory() as s:
        a = qa_history.record(s, question="身份证在哪里", answer="书房", sources=[{"raw": "原话", "doc_ids": []}], documents=[], question_vec=[0.] * 512)
        qa_history.record(s, question="花了多少", answer="38元", sources=[], documents=[], question_vec=[0.] * 512)
        s.commit(); history_id = a.id
    result = c.get("/api/qa-history?q=书房&limit=1").json()
    assert result["total"] == 1 and result["items"][0]["id"] == history_id
    assert c.get(f"/api/workspace/history/{history_id}").json()["sources"][0]["raw"] == "原话"
    assert c.get("/api/qa-history?limit=0").status_code == 422
    assert c.get("/api/qa-history?offset=-1").status_code == 422
    assert c.get("/api/qa-history?period=invalid").status_code == 422
    c.delete("/api/qa-history")
    assert c.get(f"/api/workspace/history/{history_id}").status_code == 404


def test_spending_uses_existing_stats_excludes_deleted_and_keeps_currencies(workspace):
    c, factory, root = workspace
    with factory() as s:
        now = datetime.now(timezone.utc)
        for i, amount, currency, deleted in [(1, 38, "CNY", False), (2, 5, "USD", False), (3, 99, "CNY", True)]:
            raw = store.insert_raw(s, "text", content=f"花了{amount}"); raw.deleted_at = now if deleted else None; s.flush()
            s.add(Fact(id=str(i), raw_id=raw.id, type="consumption", summary=raw.content, fields={"amount": amount, "currency": currency}, valid_from=now))
        s.commit()
    result = c.get("/api/workspace/spending").json()
    assert result["totals"] == {"CNY": 38, "USD": 5}
    assert result["count"] == 2 and all(r["raw_id"] for r in result["rows"])
    assert c.get("/api/workspace/spending?period=invalid").status_code == 422


def test_query_fallback_sources_have_local_record_and_file_links(workspace, monkeypatch):
    from app.core import answer, llm
    from app.models.schemas import QueryResponse, Source
    c, factory, root = workspace
    r = c.post("/api/reminders", json={"summary": "带上报告"}).json()
    doc_id = add_file(factory, root, raw_id=r["raw_id"])
    monkeypatch.setattr(answer, "answer", lambda *args: QueryResponse(answer="检索到的原文",
        sources=[Source(fact_id=r["fact_id"], summary="带上报告")]))
    monkeypatch.setattr(llm, "embed", lambda *args, **kw: [0.] * 512)
    result = c.post("/api/query", json={"question": "报告"}).json()
    assert result["sources"][0]["raw_id"] == r["raw_id"]
    assert result["sources"][0]["doc_ids"] == [doc_id]
    history = c.get("/api/qa-history").json()["items"][0]
    assert c.get(f'/api/workspace/history/{history["id"]}').json()["sources"][0]["doc_ids"] == [doc_id]


def test_explicit_reminder_uses_existing_local_vector_and_material_indexes(workspace, monkeypatch):
    from app.api.reminders import _index_created_reminder
    from app.core import llm
    from app.models.db import VecMap, Material
    c, factory, root = workspace
    r = c.post("/api/reminders", json={"summary": "按时带上报告"}).json()
    monkeypatch.setattr(llm, "embed", lambda *args, **kw: [0.1] * 512)
    settings = Settings(_env_file=None, data_dir=root, embedding_model="BAAI/bge-small-zh-v1.5")
    _index_created_reminder(settings, r["raw_id"], r["fact_id"])
    _index_created_reminder(settings, r["raw_id"], r["fact_id"])
    with factory() as s:
        assert len(s.scalars(select(VecMap).where(VecMap.fact_id == r["fact_id"])).all()) == 1
        assert len(s.scalars(select(Material).where(Material.source_id == r["raw_id"])).all()) == 1
