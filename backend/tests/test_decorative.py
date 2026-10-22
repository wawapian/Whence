"""装饰小图治理（V2）：保守启发式 + 只降级不丢弃 + 共用路径删除安全。"""
from io import BytesIO

from PIL import Image as PILImage
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.api.files import is_likely_decorative
from app.config import Settings
from app.core import trash
from app.memory import documents, store
from app.models.db import Document, Image, init_db


def _png_bytes(width, height, *, color=(30, 30, 30), second=None) -> bytes:
    img = PILImage.new("RGB", (width, height), color)
    if second:  # 覆盖 1/4 像素（模拟小图案/边缘）
        px = img.load()
        for y in range(height // 2):
            for x in range(width // 2):
                px[x, y] = second
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _noise_png(width=16, height=16) -> bytes:
    img = PILImage.new("RGB", (width, height), (0, 0, 0))
    px = img.load()
    for y in range(height):
        for x in range(width):
            px[x, y] = ((x * 37) % 256, (y * 53) % 256, ((x + y) * 29) % 256)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_is_likely_decorative_conservative():
    assert is_likely_decorative(_png_bytes(16, 16)) is True            # 极小均匀色块
    assert is_likely_decorative(_png_bytes(64, 64)) is False           # 面积超阈值：保住
    assert is_likely_decorative(_png_bytes(16, 16, second=(255, 255, 255))) is False  # 有 1/4 图案
    assert is_likely_decorative(_noise_png()) is False                 # 多色细节：保住
    assert is_likely_decorative(b"not an image") is False              # 解码失败：保住


def _run_doc_with_two_images(tmp_path, monkeypatch, allow_cloud=1):
    from app.api import files as files_api
    from app.core import cloud, llm
    from app.core.ingest import pipeline as pipeline_mod
    from app.core.ingest.models import ImageRef, ParsedDocument, TextChunk
    from app.memory import documents as documents_mod

    settings = Settings(data_dir=tmp_path, embedding_model="BAAI/bge-small-zh-v1.5",
                        vlm_model="qwen3-vl-plus")
    monkeypatch.setattr(llm, "embed", lambda text, model=None: [0.1] * 512)
    described = []
    monkeypatch.setattr(cloud, "describe_image",
                        lambda c, m, blob, q: described.append(blob) or "真实图描述")
    monkeypatch.setattr(cloud, "summarize", lambda c, m, text: "摘要")

    decorative_blob = _png_bytes(16, 16)
    normal_blob = _noise_png(40, 40)  # 面积 1600 > 32×32：不判装饰
    fake = pipeline_mod.IngestResult(
        parsed=ParsedDocument(
            kind="word", title="t", full_text="正文", headings=[], blocks=[],
            engine_used="python-docx",
            images=[
                ImageRef(blob=decorative_blob, page=1, position=None, ocr_text=""),
                ImageRef(blob=normal_blob, page=1, position=None, ocr_text=""),
            ],
        ),
        chunks=[TextChunk("正文", None, None, "正文", 0)],
    )
    monkeypatch.setattr(pipeline_mod, "ingest", lambda *a, **k: fake)

    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    init_db(engine, dim=512)
    with Session(engine) as s:
        doc = documents_mod.insert_document(
            s, kind="word", title="a.docx", file_path="files/a.docx",
            status="processing", allow_cloud=allow_cloud,
        )
        s.commit()
        doc_id = doc.id

    files_api.process_document(settings, doc_id, "files/a.docx", "auto", allow_cloud, 0)
    return engine, doc_id, described, normal_blob


def test_document_decorative_image_downgraded_not_discarded(tmp_path, monkeypatch):
    from app.memory.documents import visible_images

    engine, doc_id, described, normal_blob = _run_doc_with_two_images(tmp_path, monkeypatch)

    with Session(engine) as s:
        rows = s.query(Image).order_by(Image.created_at).all()
        assert len(rows) == 2  # 两张都建了行（永不丢弃）
        decorative = [r for r in rows if r.decorative == 1]
        normal = [r for r in rows if r.decorative == 0]
        assert len(decorative) == 1 and len(normal) == 1
        assert decorative[0].caption is None                                  # 装饰图不自动描述
        assert decorative[0].file_path and (tmp_path / decorative[0].file_path).exists()  # 文件保留
        assert normal[0].caption == "真实图描述"
        assert described == [normal_blob]                                     # VLM 只看了正常图
        assert [i.id for i in visible_images(s, [doc_id])] == [normal[0].id]  # 清单只含正常图


def test_is_likely_decorative_respects_off_switch(tmp_path, monkeypatch):
    """IMAGE_FILTER=off 时不判装饰（全部正常走描述/清单）。"""
    from app.api import files as files_api
    from app.core import cloud, llm
    from app.core.ingest import pipeline as pipeline_mod
    from app.core.ingest.models import ImageRef, ParsedDocument, TextChunk
    from app.memory import documents as documents_mod

    settings = Settings(data_dir=tmp_path, embedding_model="BAAI/bge-small-zh-v1.5",
                        vlm_model="qwen3-vl-plus", image_filter="off")
    monkeypatch.setattr(llm, "embed", lambda text, model=None: [0.1] * 512)
    monkeypatch.setattr(cloud, "describe_image", lambda c, m, blob, q: "描述")
    monkeypatch.setattr(cloud, "summarize", lambda c, m, text: "摘要")

    fake = pipeline_mod.IngestResult(
        parsed=ParsedDocument(
            kind="word", title="t", full_text="正文", headings=[], blocks=[],
            engine_used="python-docx",
            images=[ImageRef(blob=_png_bytes(16, 16), page=1, position=None, ocr_text="")],
        ),
        chunks=[TextChunk("正文", None, None, "正文", 0)],
    )
    monkeypatch.setattr(pipeline_mod, "ingest", lambda *a, **k: fake)

    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    init_db(engine, dim=512)
    with Session(engine) as s:
        doc = documents_mod.insert_document(
            s, kind="word", title="a.docx", file_path="files/a.docx",
            status="processing", allow_cloud=1,
        )
        s.commit()
        doc_id = doc.id

    files_api.process_document(settings, doc_id, "files/a.docx", "auto", 1, 0)

    with Session(engine) as s:
        img = s.query(Image).one()
        assert img.decorative == 0
        assert img.caption == "描述"


def test_purge_document_shared_path_keeps_file_for_live_record(tmp_path):
    """共用路径安全：单独彻底删除文档（记录仍活着）→ 记录的原文件不得被删。"""
    files_dir = tmp_path / "files"
    files_dir.mkdir()
    (files_dir / "x.jpg").write_bytes(b"\xff\xd8\xff\xe0fake")
    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    init_db(engine, dim=512)
    with Session(engine) as s:
        raw = store.insert_raw(s, kind="photo", file_path="files/x.jpg")  # 活记录
        doc = documents.insert_document(
            s, kind="photo", title="x.jpg", file_path="files/x.jpg",
            status="done", raw_id=raw.id,
        )
        s.flush()
        s.add(Image(id="img1", document_id=doc.id, file_path="files/x.jpg"))  # 与记录共用路径
        s.commit()
        doc_id = doc.id
        doc.deleted_at = store.utcnow()  # 只有文档进回收站（记录仍活）
        s.commit()

        assert trash.purge_document(s, doc_id) is True
        s.commit()
        assert s.query(Image).count() == 0
        assert s.get(Document, doc_id) is None
    assert (files_dir / "x.jpg").exists()  # 活记录的附件文件仍在


def test_purge_group_with_shared_path_deletes_file(tmp_path):
    """整组都在回收站 → 彻底删除文档连带记录，共用路径文件一并消失。"""
    files_dir = tmp_path / "files"
    files_dir.mkdir()
    (files_dir / "x.jpg").write_bytes(b"\xff\xd8\xff\xe0fake")
    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    init_db(engine, dim=512)
    with Session(engine) as s:
        raw = store.insert_raw(s, kind="photo", file_path="files/x.jpg")
        doc = documents.insert_document(
            s, kind="photo", title="x.jpg", file_path="files/x.jpg",
            status="done", raw_id=raw.id,
        )
        s.flush()
        s.add(Image(id="img1", document_id=doc.id, file_path="files/x.jpg"))
        s.commit()
        doc_id = doc.id
        doc.deleted_at = store.utcnow()
        raw.deleted_at = store.utcnow()
        s.commit()

        assert trash.purge_document(s, doc_id) is True
        s.commit()
        assert s.query(Image).count() == 0
    assert not (files_dir / "x.jpg").exists()
