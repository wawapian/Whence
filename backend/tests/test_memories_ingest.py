"""记录上传文档/图片 → 复用完整 ingestion（建 Document + 分块索引）。"""
import sqlite3
import time

from docx import Document as DocxDocument
from fastapi.testclient import TestClient

from app import deps
from app.config import Settings
from app.core import cloud, extraction, llm, parsing
from app.main import create_app
from app.models.schemas import ExtractedFact


def _make_docx(tmp_path):
    p = tmp_path / "f.docx"
    d = DocxDocument()
    d.add_paragraph("查重报告 相似度 15%")
    d.save(str(p))
    return p


def test_memories_upload_indexes_document(tmp_path, monkeypatch):
    docx = _make_docx(tmp_path)
    app = create_app()
    app.dependency_overrides[deps.get_settings] = lambda: Settings(data_dir=tmp_path)
    monkeypatch.setattr(
        extraction, "extract",
        lambda client, model, text: [ExtractedFact(type="other", summary="查重报告", confidence=0.9)],
    )
    monkeypatch.setattr(llm, "embed", lambda q, model=None: [0.1] * 1024)
    monkeypatch.setattr(parsing, "parse_document", lambda p: "查重报告 相似度 15%")

    client = TestClient(app)
    with open(docx, "rb") as f:
        resp = client.post(
            "/api/memories",
            files={"file": ("f.docx", f, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
        )
    assert resp.status_code == 200

    db = tmp_path / "mem.db"
    done = chunks = materials = 0
    for _ in range(100):
        con = sqlite3.connect(db)
        done = con.execute("SELECT count(*) FROM documents WHERE status='done'").fetchone()[0]
        chunks = con.execute("SELECT count(*) FROM chunks").fetchone()[0]
        materials = con.execute("SELECT count(*) FROM materials").fetchone()[0]
        con.close()
        # 材料是后台任务的收尾步：等到它才算整条跑完（否则测试退出时后台线程还在跑，
        # monkeypatch 撤销后会触发真实嵌入/联网）
        if done >= 1 and chunks > 0 and materials >= 2:
            break
        time.sleep(0.05)

    assert done >= 1, "文档后台 ingestion 未完成"
    assert chunks > 0, "文档未分块入索引"
    assert materials >= 2, "记录+文档材料层未就绪（后台任务未跑完）"


def test_memories_checkbox_maps_to_allow_cloud(tmp_path, monkeypatch):
    """前端「可理解」勾选框：勾选(sensitive=false) → 文档 allow_cloud=1；未勾选(true) → 0。"""
    docx = _make_docx(tmp_path)
    app = create_app()
    app.dependency_overrides[deps.get_settings] = lambda: Settings(data_dir=tmp_path)
    monkeypatch.setattr(
        extraction, "extract",
        lambda client, model, text: [ExtractedFact(type="other", summary="x", confidence=0.9)],
    )
    monkeypatch.setattr(llm, "embed", lambda q, model=None: [0.1] * 1024)
    monkeypatch.setattr(parsing, "parse_document", lambda p: "查重报告 相似度 15%")
    monkeypatch.setattr(cloud, "summarize", lambda client, model, text: "摘要")

    client = TestClient(app)
    for sensitive in ("false", "true"):
        with open(docx, "rb") as f:
            resp = client.post(
                "/api/memories",
                files={"file": ("f.docx", f, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
                data={"sensitive": sensitive},
            )
        assert resp.status_code == 200

    db = tmp_path / "mem.db"
    for _ in range(100):
        con = sqlite3.connect(db)
        done = con.execute("SELECT count(*) FROM documents WHERE status='done'").fetchone()[0]
        allowed = con.execute("SELECT count(*) FROM documents WHERE allow_cloud=1").fetchone()[0]
        denied = con.execute("SELECT count(*) FROM documents WHERE allow_cloud=0").fetchone()[0]
        materials = con.execute("SELECT count(*) FROM materials").fetchone()[0]
        con.close()
        if done >= 2 and materials >= 4:  # 4 = 2 条记录材料 + 2 份文档材料（收尾步）
            break
        time.sleep(0.05)

    assert (allowed, denied) == (1, 1), f"allow_cloud 映射错误: allowed={allowed} denied={denied}"


def _list_client(tmp_path, monkeypatch):
    from app.core import llm as llm_mod
    app = create_app()
    app.dependency_overrides[deps.get_settings] = lambda: Settings(data_dir=tmp_path)
    monkeypatch.setattr(llm_mod, "embed", lambda q, model=None: [0.1] * 1024)
    return TestClient(app)


def test_memories_list_includes_doc_ids(tmp_path, monkeypatch):
    """时间线点附件需要：/api/memories 每条带该记录的未删文档 id（软删的不在内）。"""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from app.memory import documents as documents_mod
    from app.memory import store as store_mod
    from app.models.db import init_db

    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    init_db(engine, dim=1024)  # 与应用默认 embedding 维度一致（护栏会拦 512 建表）
    with Session(engine) as s:
        raw = store_mod.insert_raw(s, kind="text", content="今天上传了两份文件")
        s.flush()
        d1 = documents_mod.insert_document(
            s, kind="document", title="a.docx", file_path="files/a.docx", status="done",
            raw_id=raw.id)
        d2 = documents_mod.insert_document(
            s, kind="document", title="b.pdf", file_path="files/b.pdf", status="done",
            raw_id=raw.id)
        documents_mod.insert_document(  # 软删的不该出现在 doc_ids
            s, kind="document", title="已删.docx", file_path="files/c.docx", status="done",
            raw_id=raw.id, deleted_at=store_mod.utcnow())
        s.commit()
        raw_id, want = raw.id, {d1.id, d2.id}

    client = _list_client(tmp_path, monkeypatch)
    entries = client.get("/api/memories").json()
    entry = next(e for e in entries if e["raw_id"] == raw_id)
    assert set(entry["doc_ids"]) == want
