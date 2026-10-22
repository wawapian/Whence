"""「已删不得泄漏」：软删后 10 个读入口逐一查不到，恢复后逐一全部回来。

10 入口：`chunks.fts_search` / `chunks.vector_search` / `fts.search` / `vector.search` /
`memories.list_memories` / `export.export_md` / `answer._answer_analysis`（选表）/
`answer._describe_retrieved_images`（图片）/ `reminders.reminders`（提醒）/
`answer._doc_fact_summaries`（总结身份材料——已删记录既不可见、也不该继续影响打分）；
外加新 helper `documents.list_documents`。
另断言「只排已删、不误伤未删」——过滤不得把可见对象一起藏掉。
"""
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app import deps
from app.api import export, memories, reminders
from app.config import Settings
from app.core import answer, cloud, stats, trash
from app.main import create_app
from app.memory import chunks as chunk_store
from app.memory import documents, fts, store, vector
from app.models.db import init_db
from app.models.schemas import ExtractedFact

EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"  # 512 维，与下面索引向量一致
VEC = [0.1] * 512


def _session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    init_db(engine, dim=512)
    return Session(engine)


def _seed_document(session, tmp_path, title="成绩总表", text="成绩总表 学分",
                   vec=None, allow_cloud=0):
    """文档 + chunk（含 chunk_fts/chunk_vec/vec_rowid）+ doc_tables + images（假图落盘）。"""
    doc = documents.insert_document(
        session, kind="word", status="done", title=title, allow_cloud=allow_cloud
    )
    session.flush()
    chunk = documents.insert_chunk(
        session, document_id=doc.id, seq=0, text=text, parent_text=text
    )
    session.flush()
    chunk_store.index_chunk(session, chunk.id, vec or VEC, text)
    documents.insert_table(session, document_id=doc.id, headers="[]", rows="[]")
    documents.insert_image(session, document_id=doc.id, file_path="images/i.png", page=1)
    images_dir = tmp_path / "images"
    images_dir.mkdir(exist_ok=True)
    (images_dir / "i.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
    session.commit()
    return doc, chunk


def _seed_record(session, content="身份证放在书房抽屉", vec=None, is_reminder=False):
    """记录：raw + fact + memory_fts/memory_vec（is_reminder=True 时同属提醒入口）。"""
    raw = store.insert_raw(session, kind="text", content=content)
    session.flush()
    fact = store.insert_fact(
        session,
        ExtractedFact(type="other", summary=content, confidence=0.9, is_reminder=is_reminder),
        raw.id,
    )
    session.flush()
    fts.index_fact(session, fact.id, content)
    vector.index_fact(session, fact.id, vec or VEC)
    session.commit()
    return raw, fact


def _reminder_ids(session):
    got = reminders.reminders(session)
    return [r["fact_id"] for r in got["due"] + got["pending"]]


def test_deleted_document_leaks_nowhere(tmp_path):
    s = _session(tmp_path)
    doc, chunk = _seed_document(s, tmp_path)

    # 基线：软删前每个读入口都看得到（证明确实是「过滤」而非查询本身失效）
    assert chunk_store.fts_search(s, "成绩总表", k=5) == [chunk.id]
    assert chunk_store.vector_search(s, VEC, k=5) == [chunk.id]
    assert [d.id for d in documents.list_documents(s)] == [doc.id]
    assert [t.id for t in documents.visible_doc_tables(s, [doc.id])]
    assert [i.id for i in documents.visible_images(s, [doc.id])]

    trash.soft_delete_document(s, doc.id)
    s.commit()

    assert chunk_store.fts_search(s, "成绩总表", k=5) == []
    assert chunk_store.vector_search(s, VEC, k=5) == []
    assert documents.list_documents(s) == []
    assert documents.visible_doc_tables(s, [doc.id]) == []
    assert documents.visible_images(s, [doc.id]) == []

    trash.restore_document(s, doc.id)
    s.commit()

    assert chunk.id in chunk_store.fts_search(s, "成绩总表", k=5)
    assert chunk.id in chunk_store.vector_search(s, VEC, k=5)
    assert [d.id for d in documents.list_documents(s)] == [doc.id]
    assert [t.id for t in documents.visible_doc_tables(s, [doc.id])]
    assert [i.id for i in documents.visible_images(s, [doc.id])]


def test_deleted_record_leaks_nowhere(tmp_path):
    s = _session(tmp_path)
    raw, fact = _seed_record(s, is_reminder=True)

    # 基线：软删前每个读入口都看得到
    assert fts.search(s, "身份证", k=5) == [fact.id]
    assert vector.search(s, VEC, k=5) == [fact.id]
    assert [m["fact_id"] for m in memories.list_memories(s)] == [fact.id]
    assert "身份证" in export.export_md(s).body.decode()
    assert _reminder_ids(s) == [fact.id]

    trash.soft_delete_record(s, raw.id)
    s.commit()

    assert fts.search(s, "身份证", k=5) == []
    assert vector.search(s, VEC, k=5) == []
    assert memories.list_memories(s) == []
    assert "身份证" not in export.export_md(s).body.decode()
    assert _reminder_ids(s) == []

    trash.restore_record(s, raw.id)
    s.commit()

    assert fts.search(s, "身份证", k=5) == [fact.id]
    assert vector.search(s, VEC, k=5) == [fact.id]
    assert [m["fact_id"] for m in memories.list_memories(s)] == [fact.id]
    assert "身份证" in export.export_md(s).body.decode()
    assert _reminder_ids(s) == [fact.id]


def test_endpoints_stop_leaking_deleted_record(tmp_path):
    """API 层真实读路径：GET /api/memories、/api/export、/api/reminders 软删后消失、恢复后回来。"""
    settings = Settings(data_dir=tmp_path, embedding_model=EMBEDDING_MODEL, rerank_engine="off")
    session = deps._session_factory(settings)()
    raw, fact = _seed_record(session, is_reminder=True)

    app = create_app()
    app.dependency_overrides[deps.get_settings] = lambda: settings
    client = TestClient(app)

    def _reminder_body():
        body = client.get("/api/reminders").json()
        return body["due"] + body["pending"]

    assert [m["fact_id"] for m in client.get("/api/memories").json()] == [fact.id]
    assert "身份证" in client.get("/api/export").text
    assert [r["fact_id"] for r in _reminder_body()] == [fact.id]

    trash.soft_delete_record(session, raw.id)
    session.commit()

    assert client.get("/api/memories").json() == []
    assert "身份证" not in client.get("/api/export").text
    assert _reminder_body() == []

    trash.restore_record(session, raw.id)
    session.commit()

    assert [m["fact_id"] for m in client.get("/api/memories").json()] == [fact.id]
    assert "身份证" in client.get("/api/export").text
    assert [r["fact_id"] for r in _reminder_body()] == [fact.id]


def test_answer_paths_skip_deleted_document(tmp_path, monkeypatch):
    """统计选表（_answer_analysis）与图片描述（_describe_retrieved_images）均不过目已删文档。"""
    s = _session(tmp_path)
    doc, chunk = _seed_document(s, tmp_path, allow_cloud=1)
    trash.soft_delete_document(s, doc.id)
    s.commit()

    # 选表：已删文档的 doc_tables 不得进入候选 → 直接判为「无可统计的表格」
    monkeypatch.setattr(stats, "extract_filter", lambda *a, **k: None)
    resp, reason = answer._answer_analysis(
        s, "学分少于1的课程", Settings(data_dir=tmp_path), client=None, chunk_ids=[chunk.id]
    )
    assert resp is None
    assert reason == "no_tables"

    # 图片：force 路径（无命中时遍历允许上云的文档）也不得描述已删文档的图片
    calls = []
    monkeypatch.setattr(cloud, "describe_image", lambda *a, **k: calls.append(a) or "图片描述")
    settings = Settings(data_dir=tmp_path, vlm_model="qwen3-vl-plus")
    assert answer._describe_retrieved_images(s, [], "图里有什么", settings, force=True) == []
    assert calls == []


def test_deleted_record_stops_feeding_summarize_identity(tmp_path):
    """已删记录不再作为「总结选哪份文档」的身份材料（_doc_fact_summaries 直调）。"""
    s = _session(tmp_path)
    doc = documents.insert_document(
        s, kind="word", status="done", title="机器人课程的论文", file_path="files/paper.docx"
    )
    s.flush()
    raw = store.insert_raw(s, kind="document", file_path="files/paper.docx")
    s.flush()
    store.insert_fact(
        s,
        ExtractedFact(type="other", summary="机器人课程的大作业是一篇论文", confidence=0.9),
        raw.id,
    )
    s.commit()

    assert answer._doc_fact_summaries(s, doc) == ["机器人课程的大作业是一篇论文"]

    trash.soft_delete_record(s, raw.id)
    s.commit()

    assert answer._doc_fact_summaries(s, doc) == []

    trash.restore_record(s, raw.id)
    s.commit()

    assert answer._doc_fact_summaries(s, doc) == ["机器人课程的大作业是一篇论文"]


def test_deleted_object_does_not_hide_visible_ones(tmp_path):
    """只排已删：删除一篇文档/一条记录，不得遮蔽另一篇/另一条。"""
    s = _session(tmp_path)
    keep_doc, keep_chunk = _seed_document(s, tmp_path, title="实验报告", text="实验报告 结论")
    drop_doc, drop_chunk = _seed_document(
        s, tmp_path, title="成绩总表", text="成绩总表 学分", vec=[0.9] * 512
    )
    _, keep_fact = _seed_record(s, content="今天买了鸡蛋")
    drop_raw, drop_fact = _seed_record(s, content="身份证放在书房抽屉", vec=[0.9] * 512)

    assert {d.id for d in documents.list_documents(s)} == {keep_doc.id, drop_doc.id}

    trash.soft_delete_document(s, drop_doc.id)
    trash.soft_delete_record(s, drop_raw.id)
    s.commit()

    assert chunk_store.fts_search(s, "成绩总表", k=5) == []
    assert chunk_store.fts_search(s, "实验报告", k=5) == [keep_chunk.id]
    assert drop_chunk.id not in chunk_store.vector_search(s, VEC, k=5)
    assert keep_chunk.id in chunk_store.vector_search(s, VEC, k=5)
    assert [d.id for d in documents.list_documents(s)] == [keep_doc.id]

    assert fts.search(s, "身份证", k=5) == []
    assert fts.search(s, "买了鸡蛋", k=5) == [keep_fact.id]
    assert drop_fact.id not in vector.search(s, VEC, k=5)
    assert keep_fact.id in vector.search(s, VEC, k=5)
    assert [m["fact_id"] for m in memories.list_memories(s)] == [keep_fact.id]
    text = export.export_md(s).body.decode()
    assert "今天买了鸡蛋" in text and "身份证" not in text
