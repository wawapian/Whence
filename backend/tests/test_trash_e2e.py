"""回收站端到端：完整 HTTP 面走通「可见 → 软删 → 回收站可见 → 恢复 → 再删 → 彻底删除」。

数据由 Session 直接种入（文档：chunk + chunk_fts/chunk_vec 索引行 + doc_tables + 磁盘文件；
记录：fact + memory_fts/memory_vec 索引行 + 附件文件），随后一律经 TestClient 的 API 动作与断言。
不用 `with TestClient(...)`：避免触发 lifespan 打到真实 data_dir。
"""
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from app import deps
from app.config import Settings
from app.main import create_app
from app.memory import chunks as chunk_store
from app.memory import documents, fts, store, vector
from app.models.db import Chunk, DocTable, Document, Fact, RawEvidence, init_db
from app.models.schemas import ExtractedFact

EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"  # 512 维，与下面索引向量一致
VEC = [0.1] * 512


def _client(tmp_path) -> TestClient:
    app = create_app()
    app.dependency_overrides[deps.get_settings] = lambda: Settings(
        data_dir=tmp_path, embedding_model=EMBEDDING_MODEL
    )
    return TestClient(app)


def _session(tmp_path) -> Session:
    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    init_db(engine, dim=512)
    return Session(engine)


def _seed_document(tmp_path, title="成绩总表") -> tuple[str, str]:
    """文档 + chunk（含 chunk_fts/chunk_vec 索引行）+ doc_tables + 磁盘文件。"""
    files_dir = tmp_path / "files"
    files_dir.mkdir(exist_ok=True)
    (files_dir / "grade.pdf").write_bytes(b"%PDF-fake")
    body = "成绩总表 体育课 90分"
    with _session(tmp_path) as s:
        doc = documents.insert_document(
            s, kind="pdf", title=title, file_path="files/grade.pdf",
            parsed_text=body, status="done",
        )
        s.flush()
        chunk = documents.insert_chunk(
            s, document_id=doc.id, seq=0, text=body, parent_text=body
        )
        s.flush()
        chunk_store.index_chunk(s, chunk.id, VEC, body)
        documents.insert_table(
            s, document_id=doc.id, headers='["课程", "学分"]', rows='[["体育课", "0.5"]]'
        )
        s.commit()
        return doc.id, chunk.id


def _seed_record(tmp_path, content="身份证放在书房抽屉") -> tuple[str, str]:
    """记录：raw（带附件文件）+ fact + memory_fts/memory_vec 索引行。"""
    files_dir = tmp_path / "files"
    files_dir.mkdir(exist_ok=True)
    (files_dir / "shot.jpg").write_bytes(b"\xff\xd8\xff\xe0fake")
    with _session(tmp_path) as s:
        raw = store.insert_raw(
            s, kind="photo", content=content, file_path="files/shot.jpg"
        )
        s.flush()
        fact = store.insert_fact(
            s, ExtractedFact(type="other", summary=content, confidence=0.9), raw.id
        )
        s.flush()
        fts.index_fact(s, fact.id, content)
        vector.index_fact(s, fact.id, VEC)
        s.commit()
        return raw.id, fact.id


def test_document_trash_e2e_over_http(tmp_path):
    client = _client(tmp_path)
    doc_id, chunk_id = _seed_document(tmp_path)

    # ① 删前：文档区可见、检索命中（基线，证明确实是「过滤」而非查询失效）
    listed = client.get("/api/documents").json()["documents"]
    assert [d["id"] for d in listed] == [doc_id]
    assert listed[0]["title"] == "成绩总表"
    with _session(tmp_path) as s:
        assert chunk_store.fts_search(s, "成绩总表", k=5) == [chunk_id]

    # ② 软删 → 204（重复删 404：已在回收站）；文档区与检索入口都查不到
    assert client.delete(f"/api/documents/{doc_id}").status_code == 204
    assert client.delete(f"/api/documents/{doc_id}").status_code == 404
    assert client.get("/api/documents").json()["documents"] == []
    with _session(tmp_path) as s:
        assert chunk_store.fts_search(s, "成绩总表", k=5) == []

    # ③ 回收站确认：看得到它，且还剩 30 天
    trash_body = client.get("/api/trash").json()
    assert [d["id"] for d in trash_body["documents"]] == [doc_id]
    assert trash_body["documents"][0]["title"] == "成绩总表"
    assert trash_body["documents"][0]["days_left"] == 30

    # ④ 恢复 → 全部回来（文档区 + 检索）
    assert client.post(f"/api/trash/documents/{doc_id}/restore").status_code == 200
    assert client.get("/api/trash").json()["documents"] == []
    assert [d["id"] for d in client.get("/api/documents").json()["documents"]] == [doc_id]
    with _session(tmp_path) as s:
        assert chunk_store.fts_search(s, "成绩总表", k=5) == [chunk_id]

    # ⑤ 再删 → 彻底删除 → 204（重复 404）；各表行与磁盘文件全消失
    assert client.delete(f"/api/documents/{doc_id}").status_code == 204
    assert client.delete(f"/api/trash/documents/{doc_id}").status_code == 204
    assert client.delete(f"/api/trash/documents/{doc_id}").status_code == 404
    assert client.get("/api/documents").json()["documents"] == []
    assert client.get("/api/trash").json()["documents"] == []
    assert not (tmp_path / "files" / "grade.pdf").exists()
    with _session(tmp_path) as s:
        assert s.query(Document).count() == 0
        assert s.query(Chunk).count() == 0
        assert s.query(DocTable).count() == 0
        assert s.execute(text("SELECT count(*) FROM chunk_fts")).scalar() == 0
        assert s.execute(text("SELECT count(*) FROM chunk_vec")).scalar() == 0


def test_record_trash_e2e_over_http(tmp_path):
    client = _client(tmp_path)
    raw_id, fact_id = _seed_record(tmp_path)

    # ① 删前：时间线可见、检索命中
    assert [m["raw_id"] for m in client.get("/api/memories").json()] == [raw_id]
    with _session(tmp_path) as s:
        assert fts.search(s, "身份证", k=5) == [fact_id]

    # ② 软删 → 204；时间线与检索入口都查不到
    assert client.delete(f"/api/memories/{raw_id}").status_code == 204
    assert client.get("/api/memories").json() == []
    with _session(tmp_path) as s:
        assert fts.search(s, "身份证", k=5) == []

    # ③ 回收站确认：看得到它（原话 + 还剩 30 天）→ 恢复 → 全回来
    trash_body = client.get("/api/trash").json()
    assert [r["id"] for r in trash_body["records"]] == [raw_id]
    assert trash_body["records"][0]["text"] == "身份证放在书房抽屉"
    assert trash_body["records"][0]["days_left"] == 30
    assert client.post(f"/api/trash/records/{raw_id}/restore").status_code == 200
    assert client.get("/api/trash").json()["records"] == []
    assert [m["raw_id"] for m in client.get("/api/memories").json()] == [raw_id]

    # ④ 再删 → 彻底删除 → 记录/事实/各索引表行与附件文件全消失
    assert client.delete(f"/api/memories/{raw_id}").status_code == 204
    assert client.delete(f"/api/trash/records/{raw_id}").status_code == 204
    assert client.get("/api/trash").json()["records"] == []
    assert not (tmp_path / "files" / "shot.jpg").exists()
    with _session(tmp_path) as s:
        assert s.query(RawEvidence).count() == 0
        assert s.query(Fact).count() == 0
        assert s.execute(text("SELECT count(*) FROM memory_fts")).scalar() == 0
        assert s.execute(text("SELECT count(*) FROM memory_vec")).scalar() == 0
        assert s.execute(text("SELECT count(*) FROM vec_map")).scalar() == 0
