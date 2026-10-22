"""回收站 API：软删 / 文档列表 / 回收站列表 / 恢复 / 彻底删除。"""
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app import deps
from app.config import Settings
from app.core import extraction, llm, queue
from app.main import create_app
from app.memory import documents, store
from app.models.db import Document, RawEvidence, init_db
from app.models.schemas import ExtractedFact


def _client(tmp_path, monkeypatch):
    app = create_app()
    app.dependency_overrides[deps.get_settings] = lambda: Settings(
        data_dir=tmp_path, embedding_model="BAAI/bge-small-zh-v1.5"
    )

    def fake_extract(client, model, text):
        return [ExtractedFact(
            type="item_location", subject_key="身份证:位置",
            summary="身份证放书房", confidence=0.9,
        )]

    monkeypatch.setattr(extraction, "extract", fake_extract)
    monkeypatch.setattr(llm, "embed", lambda text, model=None: [0.1] * 512)

    def _submit(fn, *args, **kwargs):  # 记录整理内联执行（accept-fast 的整理在后台）
        lane = kwargs.pop("lane", "doc")  # 车道参数由队列消费，不进任务函数
        if lane == "record":
            return fn(*args, **kwargs) or "test-inline"
        return "test-noop"

    monkeypatch.setattr(queue, "submit", _submit)
    return TestClient(app)


def _session(tmp_path) -> Session:
    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    init_db(engine, dim=512)
    return Session(engine)


def test_delete_trash_restore_flow(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    # 记录流（accept-fast：响应即带 raw_id；整理已内联执行）
    body = client.post("/api/memories", data={"text": "身份证放在书房"}).json()
    raw_id = body["raw_id"]
    assert client.delete(f"/api/memories/{raw_id}").status_code == 204
    assert client.delete(f"/api/memories/{raw_id}").status_code == 404
    trash_body = client.get("/api/trash").json()
    assert any(r["id"] == raw_id for r in trash_body["records"])
    assert client.post(f"/api/trash/records/{raw_id}/restore").status_code == 200
    assert client.get("/api/trash").json()["records"] == []
    assert [m["raw_id"] for m in client.get("/api/memories").json()] == [raw_id]  # 恢复后回到可见


def test_reupload_same_file_while_old_in_trash(tmp_path, monkeypatch):
    """检查题：回收站未清时重传同一份文件——新旧互不干扰；删旧不碰新；恢复旧=双份（如实记录）。

    期望行为（设计内）：
    1. 重传 = 全新 uuid 一套（raw/文件路径/文档独立），列表与检索只见新的；
    2. 彻底删除旧副本：只删旧文件与旧行，新档毫发无损；
    3. 恢复旧副本（记录+文档各自恢复）：同内容两份同时可见——已知行为，不是 bug。
    """
    import io

    client = _client(tmp_path, monkeypatch)

    # ① 第一次上传 → 进回收站（不彻底删除）
    r1 = client.post(
        "/api/memories",
        files={"file": ("合同.pdf", io.BytesIO(b"same-content"), "application/pdf")},
    )
    assert r1.status_code == 200
    raw1 = r1.json()["raw_id"]
    assert client.delete(f"/api/memories/{raw1}").status_code == 204

    # ② 回收站里旧副本还在的情况下，重传同一份
    r2 = client.post(
        "/api/memories",
        files={"file": ("合同.pdf", io.BytesIO(b"same-content"), "application/pdf")},
    )
    assert r2.status_code == 200
    raw2 = r2.json()["raw_id"]
    assert raw2 != raw1

    # 独立的文件路径 + 新的可见、旧的看不见
    with _session(tmp_path) as s:
        old_path = s.get(RawEvidence, raw1).file_path
        new_path = s.get(RawEvidence, raw2).file_path
        assert old_path != new_path
        assert (tmp_path / old_path).exists() and (tmp_path / new_path).exists()
    docs = client.get("/api/documents").json()["documents"]
    assert len(docs) == 1 and docs[0]["title"] == "合同.pdf"  # 只剩新的（旧的已删）
    assert [r["id"] for r in client.get("/api/trash").json()["records"]] == [raw1]

    # ③ 恢复旧的（记录+文档各自恢复）→ 同内容两份同时可见（已知行为）
    assert client.post(f"/api/trash/records/{raw1}/restore").status_code == 200
    old_doc_id = client.get("/api/trash").json()["documents"][0]["id"]
    assert client.post(f"/api/trash/documents/{old_doc_id}/restore").status_code == 200
    assert len(client.get("/api/documents").json()["documents"]) == 2

    # ④ 再删旧的 → 彻底删除：只删旧文件与旧行，新档毫发无损
    assert client.delete(f"/api/memories/{raw1}").status_code == 204
    assert client.delete(f"/api/trash/records/{raw1}").status_code == 204
    assert not (tmp_path / old_path).exists()   # 旧文件删了
    assert (tmp_path / new_path).exists()       # 新文件没被牵连
    with _session(tmp_path) as s:
        assert s.get(RawEvidence, raw1) is None
        assert s.get(RawEvidence, raw2) is not None
    docs = client.get("/api/documents").json()["documents"]
    assert len(docs) == 1 and docs[0]["title"] == "合同.pdf"  # 新文档仍在


def test_purge_endpoint_removes_and_404s_after(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    files_dir = tmp_path / "files"
    files_dir.mkdir()
    (files_dir / "r.txt").write_bytes(b"x")
    with _session(tmp_path) as s:
        raw = store.insert_raw(s, kind="photo", content="身份证放在书房", file_path="files/r.txt")
        s.commit()
        raw_id = raw.id

    assert client.delete(f"/api/memories/{raw_id}").status_code == 204       # 先软删
    assert client.delete(f"/api/trash/records/{raw_id}").status_code == 204  # 彻底删除
    assert client.delete(f"/api/trash/records/{raw_id}").status_code == 404  # 已物理消失
    assert client.get("/api/trash").json()["records"] == []
    assert client.delete("/api/trash/records/no-such-id").status_code == 404
    assert not (files_dir / "r.txt").exists()  # 附件文件一起清
    with _session(tmp_path) as s:
        assert s.query(RawEvidence).count() == 0


def test_list_memories_includes_raw_id(tmp_path, monkeypatch):
    """前端时间线删除按钮要按 raw_id 调 DELETE；GET /api/memories 每项须携带它。"""
    client = _client(tmp_path, monkeypatch)
    body = client.post("/api/memories", data={"text": "身份证放在书房"}).json()
    raw_id = body["raw_id"]
    (item,) = client.get("/api/memories").json()
    assert {"fact_id", "type", "summary", "created_at", "raw_id", "doc_ids"} <= set(item)
    assert item["raw_id"] == raw_id


def test_documents_list_delete_restore_and_purge(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    files_dir = tmp_path / "files"
    files_dir.mkdir()
    (files_dir / "a.txt").write_bytes(b"x")
    with _session(tmp_path) as s:
        doc = documents.insert_document(
            s, kind="txt", title="购机发票", file_path="files/a.txt", status="done"
        )
        s.commit()
        doc_id = doc.id

    listed = client.get("/api/documents").json()["documents"]
    assert [d["id"] for d in listed] == [doc_id]
    assert {"id", "title", "kind", "status", "created_at", "allow_cloud"} <= set(listed[0])
    assert listed[0]["title"] == "购机发票" and listed[0]["status"] == "done"

    assert client.delete(f"/api/documents/{doc_id}").status_code == 204
    assert client.delete(f"/api/documents/{doc_id}").status_code == 404
    assert client.get("/api/documents").json()["documents"] == []  # 列表不含已删
    trash = client.get("/api/trash").json()
    assert [d["id"] for d in trash["documents"]] == [doc_id]
    assert trash["documents"][0]["days_left"] == 30

    assert client.post(f"/api/trash/documents/{doc_id}/restore").status_code == 200
    assert [d["id"] for d in client.get("/api/documents").json()["documents"]] == [doc_id]

    assert client.delete(f"/api/documents/{doc_id}").status_code == 204
    assert client.delete(f"/api/trash/documents/{doc_id}").status_code == 204
    assert client.delete(f"/api/trash/documents/{doc_id}").status_code == 404
    assert not (files_dir / "a.txt").exists()
    with _session(tmp_path) as s:
        assert s.query(Document).count() == 0


def test_get_file_404s_when_soft_deleted_and_content_returns_after_restore(tmp_path, monkeypatch):
    """已删文档不得继续可读：GET /api/files/{doc_id} 软删后 404，恢复后内容回来。"""
    client = _client(tmp_path, monkeypatch)
    files_dir = tmp_path / "files"
    files_dir.mkdir()
    (files_dir / "a.txt").write_bytes(b"x")
    with _session(tmp_path) as s:
        doc = documents.insert_document(
            s, kind="txt", title="购机发票", file_path="files/a.txt", status="done",
            parsed_text="发票正文", summary="购机发票摘要",
        )
        s.commit()
        doc_id = doc.id

    got = client.get(f"/api/files/{doc_id}")
    assert got.status_code == 200
    assert got.json()["parsed_text"] == "发票正文" and got.json()["summary"] == "购机发票摘要"

    assert client.delete(f"/api/documents/{doc_id}").status_code == 204
    assert client.get(f"/api/files/{doc_id}").status_code == 404  # 软删 → 内容不再可读
    assert client.get("/api/files/no-such-id").status_code == 404

    assert client.post(f"/api/trash/documents/{doc_id}/restore").status_code == 200
    got = client.get(f"/api/files/{doc_id}")
    assert got.status_code == 200
    assert got.json()["parsed_text"] == "发票正文"


def test_purge_live_record_404s_and_keeps_object(tmp_path, monkeypatch):
    """不在回收站（存在但未软删）→ 彻底删除 404，对象原样保留（不做隐式动作）。"""
    client = _client(tmp_path, monkeypatch)
    with _session(tmp_path) as s:
        raw = store.insert_raw(s, kind="text", content="身份证放在书房")
        s.commit()
        raw_id = raw.id

    assert client.delete(f"/api/trash/records/{raw_id}").status_code == 404
    assert client.get("/api/trash").json()["records"] == []  # 未被顺手软删
    with _session(tmp_path) as s:
        assert s.query(RawEvidence).count() == 1


def test_purge_live_document_404s_and_keeps_object(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    files_dir = tmp_path / "files"
    files_dir.mkdir()
    (files_dir / "a.txt").write_bytes(b"x")
    with _session(tmp_path) as s:
        doc = documents.insert_document(
            s, kind="txt", title="购机发票", file_path="files/a.txt", status="done"
        )
        s.commit()
        doc_id = doc.id

    assert client.delete(f"/api/trash/documents/{doc_id}").status_code == 404
    assert client.get("/api/trash").json()["documents"] == []
    assert (files_dir / "a.txt").exists()  # 文件未被删
    with _session(tmp_path) as s:
        assert s.get(Document, doc_id) is not None


def test_restore_endpoint_404s_for_non_trash_ids(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    with _session(tmp_path) as s:
        doc = documents.insert_document(s, kind="txt", title="购机发票", status="done")
        raw = store.insert_raw(s, kind="text", content="身份证放在书房")
        s.commit()
        doc_id, raw_id = doc.id, raw.id

    assert client.post(f"/api/trash/records/{raw_id}/restore").status_code == 404    # 在库但未软删
    assert client.post(f"/api/trash/documents/{doc_id}/restore").status_code == 404
    assert client.post("/api/trash/records/no-such-id/restore").status_code == 404
    assert client.post("/api/trash/documents/no-such-id/restore").status_code == 404
    assert client.get("/api/trash").json() == {"records": [], "documents": []}
    assert client.delete("/api/trash/documents/no-such-id").status_code == 404


def test_empty_trash_endpoint_clears_all_and_keeps_live(tmp_path, monkeypatch):
    """DELETE /api/trash：站内对象全清并返回计数；活对象不动；空站幂等返回 0。"""
    client = _client(tmp_path, monkeypatch)
    files_dir = tmp_path / "files"
    files_dir.mkdir()
    (files_dir / "a.txt").write_bytes(b"x")
    (files_dir / "live.txt").write_bytes(b"y")
    with _session(tmp_path) as s:
        doc = documents.insert_document(
            s, kind="txt", title="购机发票", file_path="files/a.txt", status="done"
        )
        live = documents.insert_document(
            s, kind="txt", title="活文档", file_path="files/live.txt", status="done"
        )
        raw = store.insert_raw(s, kind="text", content="身份证放在书房")
        s.commit()
        doc_id, live_id, raw_id = doc.id, live.id, raw.id

    assert client.delete(f"/api/documents/{doc_id}").status_code == 204
    assert client.delete(f"/api/memories/{raw_id}").status_code == 204

    resp = client.delete("/api/trash")
    assert resp.status_code == 200
    assert resp.json() == {"records": 1, "documents": 1}
    assert client.get("/api/trash").json() == {"records": [], "documents": []}
    assert not (files_dir / "a.txt").exists()
    assert (files_dir / "live.txt").exists()  # 活对象不受影响
    with _session(tmp_path) as s:
        assert s.get(Document, doc_id) is None
        assert s.get(Document, live_id) is not None
        assert s.query(RawEvidence).count() == 0

    assert client.delete("/api/trash").json() == {"records": 0, "documents": 0}  # 空站幂等


def test_delete_record_also_trashes_linked_document_over_http(tmp_path, monkeypatch):
    """HTTP 面联动：删记录 → 文档区同步消失、回收站两栏都出现；恢复各自独立。"""
    client = _client(tmp_path, monkeypatch)
    files_dir = tmp_path / "files"
    files_dir.mkdir()
    (files_dir / "a.txt").write_bytes(b"x")
    with _session(tmp_path) as s:
        doc = documents.insert_document(
            s, kind="txt", title="购机发票", file_path="files/a.txt", status="done"
        )
        raw = store.insert_raw(s, kind="document", file_path="files/a.txt")
        s.commit()
        doc_id, raw_id = doc.id, raw.id

    assert client.delete(f"/api/memories/{raw_id}").status_code == 204
    assert client.get("/api/documents").json()["documents"] == []  # 文档区同步消失
    trash_body = client.get("/api/trash").json()
    assert [r["id"] for r in trash_body["records"]] == [raw_id]
    assert [d["id"] for d in trash_body["documents"]] == [doc_id]

    assert client.post(f"/api/trash/records/{raw_id}/restore").status_code == 200
    assert client.get("/api/documents").json()["documents"] == []  # 恢复记录不带回文档（各自恢复）
    assert client.post(f"/api/trash/documents/{doc_id}/restore").status_code == 200
    assert [d["id"] for d in client.get("/api/documents").json()["documents"]] == [doc_id]
