import sqlite3
import time

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app import deps
from app.config import Settings
from app.core import llm
from app.core.ingest import pipeline as pipeline_mod
from app.core.ingest.models import ImageRef, ParsedDocument, TextChunk
from app.main import create_app
from app.memory import chunks


def _client(tmp_path, monkeypatch):
    app = create_app()
    app.dependency_overrides[deps.get_settings] = lambda: Settings(
        data_dir=tmp_path, embedding_model="BAAI/bge-small-zh-v1.5")
    monkeypatch.setattr(llm, "embed", lambda text, model=None: [0.1] * 512)
    fake = pipeline_mod.IngestResult(
        parsed=ParsedDocument(kind="word", title="t", full_text="正文",
                              headings=[], blocks=[], engine_used="python-docx"),
        chunks=[TextChunk("块一：学习率设为 0.001", None, None, "块一", 0),
                TextChunk("块二：优化器设为 Adam", None, None, "块二", 1)],
    )
    monkeypatch.setattr(pipeline_mod, "ingest", lambda *a, **k: fake)
    return TestClient(app)


def _wait_materials(tmp_path, n, timeout=5.0):
    """等材料层落库：材料是后台任务的收尾步——等到它才算整条跑完，避免测试退出时后台线程残留。"""
    db = tmp_path / "mem.db"
    deadline = time.time() + timeout
    while time.time() < deadline:
        con = sqlite3.connect(db)
        cnt = con.execute("SELECT count(*) FROM materials").fetchone()[0]
        con.close()
        if cnt >= n:
            return
        time.sleep(0.05)


def test_upload_document(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    files = {"file": ("a.docx", b"fake", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
    resp = client.post("/api/files", files=files, data={"allow_cloud": "0", "multimodal": "0", "ocr": "auto"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "processing"
    for _ in range(50):
        time.sleep(0.1)
        d = client.get(f"/api/files/{body['document_id']}").json()
        if d["status"] == "done":
            break
    assert d["status"] == "done"
    assert d["chunk_count"] == 2
    _wait_materials(tmp_path, 1)


def test_process_indexes_chunks(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    files = {"file": ("a.docx", b"fake", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
    resp = client.post("/api/files", files=files, data={"allow_cloud": "0", "multimodal": "0", "ocr": "auto"})
    body = resp.json()
    for _ in range(50):
        time.sleep(0.1)
        d = client.get(f"/api/files/{body['document_id']}").json()
        if d["status"] == "done":
            break
    assert d["status"] == "done"
    _wait_materials(tmp_path, 1)

    session = Session(create_engine(f"sqlite:///{tmp_path}/mem.db"))
    try:
        assert chunks.fts_search(session, "学习率") != []
    finally:
        session.close()


def _fake_ingest(monkeypatch, chunks_=None):
    """把 ingest 打桩为固定的 word 解析结果（不碰真文件）。"""
    fake = pipeline_mod.IngestResult(
        parsed=ParsedDocument(kind="word", title="t", full_text="正文",
                              headings=[], blocks=[], engine_used="python-docx"),
        chunks=chunks_ or [TextChunk("块一：学习率设为 0.001", None, None, "块一", 0)],
    )
    monkeypatch.setattr(pipeline_mod, "ingest", lambda *a, **k: fake)


def test_material_failure_does_not_mark_document_failed(tmp_path, monkeypatch):
    """F5 覆盖洞①：材料阶段失败（fail-soft）不影响已完成的文档与分块。"""
    from app.api import files as files_api
    from app.memory import documents as documents_mod, materials as materials_mod
    from app.models.db import Chunk, Document, Material, init_db

    settings = Settings(data_dir=tmp_path, embedding_model="BAAI/bge-small-zh-v1.5")
    monkeypatch.setattr(llm, "embed", lambda text, model=None: [0.1] * 512)
    _fake_ingest(monkeypatch)

    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    init_db(engine, dim=512)
    with Session(engine) as s:
        doc = documents_mod.insert_document(
            s, kind="word", title="a.docx", file_path="files/a.docx", status="processing"
        )
        s.commit()
        doc_id = doc.id

    def _boom(*a, **k):
        raise RuntimeError("material boom")

    monkeypatch.setattr(materials_mod, "upsert_for_document", _boom)
    files_api.process_document(settings, doc_id, "files/a.docx", "auto", 0, 0)  # 不抛

    with Session(engine) as s:
        assert s.get(Document, doc_id).status == "done"
        assert s.query(Chunk).filter(Chunk.document_id == doc_id).count() == 1
        assert s.query(Material).count() == 0


def test_process_document_creates_material_per_doc(tmp_path, monkeypatch):
    """F5 覆盖洞②：多附件场景——每份文档处理后各自生成材料层条目。"""
    from app.api import files as files_api
    from app.memory import documents as documents_mod
    from app.models.db import Material, init_db

    settings = Settings(data_dir=tmp_path, embedding_model="BAAI/bge-small-zh-v1.5")
    monkeypatch.setattr(llm, "embed", lambda text, model=None: [0.1] * 512)
    _fake_ingest(monkeypatch)

    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    init_db(engine, dim=512)
    with Session(engine) as s:
        ids = []
        for name in ("a.docx", "b.docx"):
            doc = documents_mod.insert_document(
                s, kind="word", title=name, file_path=f"files/{name}", status="processing"
            )
            s.flush()
            ids.append(doc.id)
        s.commit()

    for doc_id, name in zip(ids, ("a.docx", "b.docx")):
        files_api.process_document(settings, doc_id, f"files/{name}", "auto", 0, 0)

    with Session(engine) as s:
        rows = s.query(Material).all()
        assert {m.source_id for m in rows} == set(ids)
        assert all(m.source_type == "document" for m in rows)


def _fake_photo_ingest(monkeypatch, png_bytes=b"\x89PNG\r\n\x1a\nfake-bill", ocr_text="38 元"):
    """把 ingest 打桩为「随手拍照片」的解析结果（含照片本体的 ImageRef）。"""
    fake = pipeline_mod.IngestResult(
        parsed=ParsedDocument(
            kind="image", title=None, full_text=ocr_text, headings=[], blocks=[],
            engine_used="rapidocr", ocr_used=True,
            images=[ImageRef(blob=png_bytes, page=None, position="整图", ocr_text=ocr_text)],
        ),
        chunks=[TextChunk(ocr_text, None, None, ocr_text, 0)] if ocr_text else [],
    )
    monkeypatch.setattr(pipeline_mod, "ingest", lambda *a, **k: fake)


def _run_photo_doc(tmp_path, monkeypatch, allow_cloud):
    from app.api import files as files_api
    from app.core import cloud
    from app.memory import documents as documents_mod
    from app.models.db import init_db

    settings = Settings(data_dir=tmp_path, embedding_model="BAAI/bge-small-zh-v1.5",
                        vlm_model="qwen3-vl-plus")
    monkeypatch.setattr(llm, "embed", lambda text, model=None: [0.1] * 512)
    _fake_photo_ingest(monkeypatch)
    calls = []
    monkeypatch.setattr(cloud, "describe_image",
                        lambda c, m, blob, q: calls.append(q) or "一张写着 38 元的账单照片")
    monkeypatch.setattr(cloud, "summarize", lambda c, m, text: "账单摘要")  # 防云总结真调用

    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    init_db(engine, dim=512)
    with Session(engine) as s:
        doc = documents_mod.insert_document(
            s, kind="photo", title="bill.png", file_path="files/bill.png",
            status="processing", allow_cloud=allow_cloud,
        )
        s.commit()
        doc_id = doc.id

    files_api.process_document(settings, doc_id, "files/bill.png", "auto", allow_cloud, 0)
    return engine, doc_id, calls


def test_photo_document_creates_image_row_and_caption(tmp_path, monkeypatch):
    """随手拍修复：照片登记为 images 行；勾选可理解 → VLM caption + 图片描述块进索引。"""
    from app.models.db import Chunk, Document, Image

    engine, doc_id, calls = _run_photo_doc(tmp_path, monkeypatch, allow_cloud=1)

    with Session(engine) as s:
        assert s.get(Document, doc_id).status == "done"
        img = s.query(Image).one()
        assert img.document_id == doc_id
        assert img.ocr_text == "38 元"
        assert img.file_path == "files/bill.png"  # 独立照片直接引用原文件（不再复制一份）
        assert not (tmp_path / "images").exists() or not any((tmp_path / "images").iterdir())
        assert img.caption == "一张写着 38 元的账单照片"
        assert len(calls) == 1                                        # 勾选 → 恰好一次 VLM
        texts = [c.text for c in s.query(Chunk).filter(Chunk.document_id == doc_id)]
        assert any("图片描述" in t and "38 元" in t for t in texts)     # caption 描述块已建


def test_photo_document_not_checked_never_calls_vlm(tmp_path, monkeypatch):
    """未勾选（allow_cloud=0）：照片登记 images 行、OCR 文本在，但绝不调 VLM。"""
    from app.models.db import Image

    engine, doc_id, calls = _run_photo_doc(tmp_path, monkeypatch, allow_cloud=0)

    with Session(engine) as s:
        img = s.query(Image).one()
        assert img.ocr_text == "38 元"
        assert img.caption is None
    assert calls == []  # describe_image 一次未被调用（记录列表为空）


def test_chunk_embeds_precomputed_before_writes(tmp_path, monkeypatch):
    """回归：分块向量先全部算完、再进短事务写库——大文档不得攥着 SQLite 写锁做嵌入。

    反例：用户实测大 Excel（362 行 → 245 块）入库期间，删除请求撞写锁 busy 超时 → 500。
    """
    from app.api import files as files_api
    from app.memory import documents as documents_mod
    from app.models.db import init_db

    settings = Settings(data_dir=tmp_path, embedding_model="BAAI/bge-small-zh-v1.5")
    events = []
    monkeypatch.setattr(llm, "embed",
                        lambda text, model=None: events.append("embed") or [0.1] * 512)
    fake = pipeline_mod.IngestResult(
        parsed=ParsedDocument(kind="word", title="t", full_text="正文",
                              headings=[], blocks=[], engine_used="python-docx"),
        chunks=[TextChunk("块一", None, None, "块一", 0), TextChunk("块二", None, None, "块二", 1)],
    )
    monkeypatch.setattr(pipeline_mod, "ingest", lambda *a, **k: fake)
    orig_insert = documents_mod.insert_chunk

    def spy_insert(*a, **k):
        events.append("insert")
        return orig_insert(*a, **k)

    monkeypatch.setattr(documents_mod, "insert_chunk", spy_insert)

    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    init_db(engine, dim=512)
    with Session(engine) as s:
        doc = documents_mod.insert_document(
            s, kind="word", title="a.docx", file_path="files/a.docx", status="processing"
        )
        s.commit()
        doc_id = doc.id

    files_api.process_document(settings, doc_id, "files/a.docx", "auto", 0, 0)

    # 首次写库前，两次嵌入必须都已经发生（此后 materials 阶段还会有嵌入，不参与断言）
    first_insert = events.index("insert")
    assert events[:first_insert] == ["embed", "embed"]


# ---------- 原件端点（/api/files/{id}/raw）----------

def _seed_raw_doc(tmp_path, *, rel_path="files/a.txt", title="报告.txt", data=b"hello world",
                  deleted=False, allow_cloud=0):
    """建一条文档行 + 落盘文件，返回 doc_id（与 _client 共用同一 data_dir/库）。"""
    from app.memory import documents as documents_mod
    from app.memory import store
    from app.models.db import init_db

    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    init_db(engine)
    with Session(engine) as s:
        doc = documents_mod.insert_document(
            s, kind="text", title=title, file_path=rel_path, status="done",
            allow_cloud=allow_cloud)
        if deleted:
            doc.deleted_at = store.utcnow()
        s.commit()
        doc_id = doc.id
    if rel_path and data is not None:
        p = tmp_path / rel_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    return doc_id


def test_raw_serves_file_with_original_name(tmp_path, monkeypatch):
    import urllib.parse

    client = _client(tmp_path, monkeypatch)
    doc_id = _seed_raw_doc(tmp_path, title="体检报告.txt")
    resp = client.get(f"/api/files/{doc_id}/raw")
    assert resp.status_code == 200
    assert resp.content == b"hello world"
    disposition = resp.headers["content-disposition"]
    assert "inline" in disposition  # 文本原件在浏览器中显示，保留明确下载入口
    assert resp.headers["content-type"].startswith("text/plain")
    assert resp.headers["x-content-type-options"] == "nosniff"
    assert "attachment" in client.get(f"/api/files/{doc_id}/raw?disposition=attachment").headers["content-disposition"]
    quoted = disposition.split("UTF-8''", 1)[1].strip('"')
    assert urllib.parse.unquote(quoted) == "体检报告.txt"  # RFC 5987 原名


def test_raw_inline_for_pdf_and_image(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    pdf_id = _seed_raw_doc(tmp_path, rel_path="files/a.pdf", title="合同.pdf", data=b"%PDF-1.4")
    png_id = _seed_raw_doc(tmp_path, rel_path="files/a.png", title="证件.png", data=b"\x89PNG")
    assert "inline" in client.get(f"/api/files/{pdf_id}/raw").headers["content-disposition"]
    assert "inline" in client.get(f"/api/files/{png_id}/raw").headers["content-disposition"]
    forced = client.get(f"/api/files/{pdf_id}/raw", params={"disposition": "attachment"})
    assert "attachment" in forced.headers["content-disposition"]


def test_raw_soft_deleted_404(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    doc_id = _seed_raw_doc(tmp_path, deleted=True)
    assert client.get(f"/api/files/{doc_id}/raw").status_code == 404


def test_raw_missing_file_404(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    doc_id = _seed_raw_doc(tmp_path, rel_path="files/gone.txt", data=None)  # 库里指向，盘上没有
    assert client.get(f"/api/files/{doc_id}/raw").status_code == 404


def test_raw_null_relpath_404(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    doc_id = _seed_raw_doc(tmp_path, rel_path=None, data=None)
    assert client.get(f"/api/files/{doc_id}/raw").status_code == 404


def test_raw_path_traversal_404(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    (tmp_path.parent / "secret.key").write_bytes(b"TOP-SECRET")
    for bad in ("../secret.key", str(tmp_path.parent / "secret.key")):
        doc_id = _seed_raw_doc(tmp_path, rel_path=bad, data=None)
        resp = client.get(f"/api/files/{doc_id}/raw")
        assert resp.status_code == 404
        assert b"TOP-SECRET" not in resp.content


# ---------- 图片原件端点（/api/images/{id}/raw）----------

def _seed_image(tmp_path, *, allow_cloud=0, deleted=False,
                doc_file_path="files/doc.pdf", doc_data=b"%PDF-1.4 doc-bytes",
                image_file_path="images/p.png", image_data=b"\x89PNG\r\n\x1a\nfakepng"):
    """建 文档+图片行 + 落盘 bytes，返回 (doc_id, image_id)。

    文档与图片用**不同路径、不同字节**（终审复现场景：doc.pdf 与 image.png 字节不同）——
    图片端点若错吐文档文件，判别性断言（图片 raw==图片字节 / content-type / 文件名）立刻失败。
    """
    from app.memory import documents as documents_mod
    from app.memory import store
    from app.models.db import init_db

    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    init_db(engine)
    with Session(engine) as s:
        doc = documents_mod.insert_document(
            s, kind="document", title="报告.pdf", file_path=doc_file_path, status="done",
            allow_cloud=allow_cloud)
        if deleted:
            doc.deleted_at = store.utcnow()
        s.flush()
        img = documents_mod.insert_image(
            s, document_id=doc.id, file_path=image_file_path, page=None, position="整图",
            ocr_text="", decorative=0)
        s.commit()
        doc_id, image_id = doc.id, img.id
    for rel_path, data in ((doc_file_path, doc_data), (image_file_path, image_data)):
        if rel_path and data is not None:
            p = tmp_path / rel_path
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
    return doc_id, image_id


def test_image_raw_serves_bytes(tmp_path, monkeypatch):
    """判别性：图片端点吐 images.file_path 的本体（而非所属文档的文件）；两侧互不串。"""
    import urllib.parse

    client = _client(tmp_path, monkeypatch)
    doc_id, image_id = _seed_image(tmp_path)
    resp = client.get(f"/api/images/{image_id}/raw")
    assert resp.status_code == 200
    assert resp.content == b"\x89PNG\r\n\x1a\nfakepng"  # 图片字节，不是 doc.pdf 字节
    assert resp.headers["content-type"] == "image/png"  # 按图片路径推，不是 doc 的 application/pdf
    disposition = resp.headers["content-disposition"]
    assert "inline" in disposition  # 按图片路径后缀（.png）判缺省
    quoted = disposition.split("UTF-8''", 1)[1].strip('"')
    assert urllib.parse.unquote(quoted) == "p.png"  # 存储名，不是文档原名「报告.pdf」
    # 互不串：文档端点取到的必须仍是文档字节
    doc_resp = client.get(f"/api/files/{doc_id}/raw")
    assert doc_resp.status_code == 200
    assert doc_resp.content == b"%PDF-1.4 doc-bytes"


def test_image_raw_when_allow_cloud_off(tmp_path, monkeypatch):
    """本地取原件与「上云闸」是两条闸：未勾选可理解照样能看。"""
    client = _client(tmp_path, monkeypatch)
    _, image_id = _seed_image(tmp_path, allow_cloud=0)
    assert client.get(f"/api/images/{image_id}/raw").status_code == 200


def test_image_raw_doc_deleted_404(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    _, image_id = _seed_image(tmp_path, deleted=True)
    assert client.get(f"/api/images/{image_id}/raw").status_code == 404


def test_image_raw_null_file_path_404(tmp_path, monkeypatch):
    """图片行 file_path 为空 → 404（即便所属文档有可读文件，也不得回退吐文档文件）。"""
    client = _client(tmp_path, monkeypatch)
    _, image_id = _seed_image(tmp_path, image_file_path=None, image_data=None)
    assert client.get(f"/api/images/{image_id}/raw").status_code == 404


def test_documents_list_includes_allow_cloud(tmp_path, monkeypatch):
    """文件库/未来界面需要：/api/documents 每条带 allow_cloud（决定"可否上云"标）。"""
    client = _client(tmp_path, monkeypatch)
    _seed_raw_doc(tmp_path, title="上云.pdf", rel_path="files/x.pdf", allow_cloud=1)
    _seed_raw_doc(tmp_path, title="本地.pdf", rel_path="files/y.pdf")
    docs = {d["title"]: d for d in client.get("/api/documents").json()["documents"]}
    assert docs["上云.pdf"]["allow_cloud"] is True
    assert docs["本地.pdf"]["allow_cloud"] is False
