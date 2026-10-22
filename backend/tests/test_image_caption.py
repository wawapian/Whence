from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session

from app.config import Settings
from app.core import llm
from app.models.db import init_db


def test_init_db_adds_caption_column_to_existing_table(tmp_path):
    """旧库（images 表无 caption 列）init_db 后补列，且重复执行幂等。"""
    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE images (id TEXT PRIMARY KEY, document_id TEXT NOT NULL, "
            "file_path TEXT, page INTEGER, position TEXT, ocr_text TEXT, created_at TEXT NOT NULL)"
        ))
    # 断开播种连接：init_db 的 sqlite-vec 加载钩子只对新建立的连接生效，
    # 复用旧连接会报 no such module: vec0
    engine.dispose()
    init_db(engine, dim=512)
    with engine.begin() as conn:
        cols = {r[1] for r in conn.execute(text("PRAGMA table_info(images)"))}
    assert "caption" in cols
    init_db(engine, dim=512)  # 幂等：不抛错


def _fake_ingest_result(image_count=1):
    from app.core.ingest.models import ImageRef, ParsedDocument, TextChunk
    from app.core.ingest.pipeline import IngestResult
    parsed = ParsedDocument(
        kind="word", title="t", full_text="正文", headings=[], blocks=[],
        engine_used="fake", ocr_used=False, needs_ocr=None,
        images=[ImageRef(blob=b"\x89PNG\r\n\x1a\nfake", page=None, position=None, ocr_text="")
                for _ in range(image_count)],
    )
    chunks = [TextChunk(text="正文块", heading_path=None, page=None, parent_text="正文块", seq=0)]
    return IngestResult(parsed=parsed, chunks=chunks)


def _run_ingest(tmp_path, monkeypatch, image_count=1, allow_cloud=1, describe=None,
                vlm_model="qwen3-vl-plus", fail_caption_embed=False):
    """建库+文档行，跑一次 process_document，返回 (读取用 Session, doc_id)。

    注意 init_db 维度必须与会话工厂一致（bge-m3 → 1024），否则 vec 表维度不匹配。
    """
    from app.api import files
    from app.memory import documents
    if describe is None:
        describe = lambda c, m, blob, q: "一张测试图：包含颜色 A 与数字 42"
    monkeypatch.setattr(files.pipeline, "ingest", lambda *a, **k: _fake_ingest_result(image_count))
    monkeypatch.setattr(files.cloud, "describe_image", describe)
    settings = Settings(data_dir=tmp_path, vlm_model=vlm_model)

    def fake_embed(q, model=None):
        # 锁定两参数形式：caption chunk 必须与正文 chunk 同 embedding 模型（防维度错配）
        assert model == settings.embedding_model, f"embed model mismatch: {model!r}"
        if fail_caption_embed and q.startswith("图片描述"):
            raise RuntimeError("embed down")
        return [0.1] * 1024

    monkeypatch.setattr(llm, "embed", fake_embed)
    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    init_db(engine, dim=1024)
    session = Session(engine)
    doc = documents.insert_document(session, kind="word", title="实验报告",
                                    file_path="files/x.docx", status="processing")
    session.commit()
    doc_id = doc.id
    session.close()
    files.process_document(settings, doc_id, "files/x.docx", "off", allow_cloud, 0)
    return Session(engine), doc_id


def test_ingest_generates_caption_and_indexes_chunk(tmp_path, monkeypatch):
    from app.memory import chunks
    from app.models.db import Image
    session, doc_id = _run_ingest(tmp_path, monkeypatch, allow_cloud=1)
    row = session.scalar(select(Image).where(Image.document_id == doc_id))
    assert row.caption and "42" in row.caption
    assert chunks.fts_search(session, "测试图", k=5)  # caption 虚拟 chunk 可检索（≥3 字才建 trigram）


def test_ingest_skips_caption_when_not_allow_cloud(tmp_path, monkeypatch):
    from app.memory import chunks
    from app.models.db import Image
    session, doc_id = _run_ingest(tmp_path, monkeypatch, allow_cloud=0)
    assert session.scalar(select(Image).where(Image.document_id == doc_id)).caption is None
    assert chunks.fts_search(session, "测试图", k=5) == []


def test_ingest_skips_caption_when_vlm_not_configured(tmp_path, monkeypatch):
    """双门控另一侧：allow_cloud=1 但未配置 vlm_model → 不生成 caption、不建 chunk。"""
    from app.memory import chunks
    from app.models.db import Image
    session, doc_id = _run_ingest(tmp_path, monkeypatch, allow_cloud=1, vlm_model="")
    assert session.scalar(select(Image).where(Image.document_id == doc_id)).caption is None
    assert chunks.fts_search(session, "测试图", k=5) == []


def test_ingest_caption_failure_skipped(tmp_path, monkeypatch):
    from app.models.db import Document, Image

    def _boom(*a, **k):
        raise RuntimeError("vlm down")

    session, doc_id = _run_ingest(tmp_path, monkeypatch, allow_cloud=1, describe=_boom)
    assert session.get(Document, doc_id).status == "done"
    assert session.scalar(select(Image).where(Image.document_id == doc_id)).caption is None


def test_ingest_caption_index_failure_does_not_fail_document(tmp_path, monkeypatch):
    """describe 之后的向量化/索引失败也只跳过该图：文档仍 done、该图部分写入已回滚、无 caption chunk。"""
    from app.models.db import Chunk, Document, Image
    session, doc_id = _run_ingest(tmp_path, monkeypatch, allow_cloud=1, fail_caption_embed=True)
    assert session.get(Document, doc_id).status == "done"
    assert session.scalar(select(Image).where(Image.document_id == doc_id)).caption is None
    # 直接查 Chunk 行（fts_search 会吞异常，不足以证明行不存在）
    assert session.scalars(
        select(Chunk).where(Chunk.document_id == doc_id, Chunk.heading_path == "图片")
    ).all() == []


def test_ingest_vlm_client_failure_does_not_fail_document(tmp_path, monkeypatch):
    """caption 阶段整体失败（如 VLM 客户端构造抛错）只跳过该阶段：文档仍 done、无 caption。"""
    from app.core import cloud
    from app.models.db import Document, Image

    def _boom(settings):
        raise RuntimeError("vlm client down")

    monkeypatch.setattr(cloud, "build_vlm_client", _boom)
    session, doc_id = _run_ingest(tmp_path, monkeypatch, allow_cloud=1)
    assert session.get(Document, doc_id).status == "done"
    assert session.scalar(select(Image).where(Image.document_id == doc_id)).caption is None


def test_ingest_caption_cap(tmp_path, monkeypatch):
    from app.models.db import Image
    session, doc_id = _run_ingest(tmp_path, monkeypatch, image_count=25, allow_cloud=1)
    captioned = session.scalars(
        select(Image).where(Image.document_id == doc_id, Image.caption.isnot(None))
    ).all()
    assert len(captioned) == 20


def test_ingested_caption_reaches_query_synthesis_contexts(tmp_path, monkeypatch):
    """端到端贯通：入库 caption 的 chunk 可被检索，并作为上下文进入合成。"""
    from app.core import answer
    session, _ = _run_ingest(tmp_path, monkeypatch,
                             describe=lambda c, m, blob, q: "彩虹色菱形图案")
    captured = []
    monkeypatch.setattr(answer, "synthesize",
                        lambda client, model, q, contexts, images=None: captured.append(contexts) or "答案")
    settings = Settings(data_dir=tmp_path, vlm_model="qwen3-vl-plus", rerank_engine="off",
                        qa_engine="legacy")
    answer.answer(session, "彩虹色菱形图案是什么", settings)
    assert captured and any("图片描述" in c and "彩虹色菱形图案" in c for c in captured[0])


def test_backfill_generates_missing_captions(tmp_path, monkeypatch):
    from scripts.caption_backfill import backfill
    from app.memory import chunks, documents
    from app.models.db import Document, Image
    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    init_db(engine, dim=512)
    session = Session(engine)
    doc = documents.insert_document(session, kind="word", status="done",
                                    title="旧文档", file_path="files/old.docx", allow_cloud=1)
    session.flush()
    img_dir = tmp_path / "images"; img_dir.mkdir()
    (img_dir / "old.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
    session.add(Image(id="img1", document_id=doc.id, file_path="images/old.png"))
    session.commit()
    monkeypatch.setattr("app.core.cloud.describe_image", lambda c, m, blob, q: "旧图描述X")
    monkeypatch.setattr(llm, "embed", lambda q, model=None: [0.1] * 512)
    stats = backfill(Settings(data_dir=tmp_path, vlm_model="qwen3-vl-plus"), session=session)
    assert stats["captioned"] == 1 and stats["remaining"] == 0
    session.expire_all()
    assert session.get(Image, "img1").caption == "旧图描述X"
    assert chunks.fts_search(session, "旧图描述", k=5)


def test_backfill_isolates_single_image_failure(tmp_path, monkeypatch):
    """一张图向量化失败只记 failed（无异常逃出循环），另一张照常补录并落 caption chunk。"""
    from scripts.caption_backfill import backfill
    from app.memory import documents
    from app.models.db import Chunk, Document, Image

    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    init_db(engine, dim=512)
    session = Session(engine)
    doc = documents.insert_document(session, kind="word", status="done",
                                    title="旧文档", file_path="files/old.docx", allow_cloud=1)
    session.flush()
    img_dir = tmp_path / "images"; img_dir.mkdir()
    (img_dir / "old.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
    # 甲/乙 进入 image_source 标注，让 embed 能精确地只对第一张图的文本炸掉
    session.add(Image(id="imgA", document_id=doc.id, file_path="images/old.png", position="甲"))
    session.add(Image(id="imgB", document_id=doc.id, file_path="images/old.png", position="乙"))
    session.commit()

    def fake_embed(q, model=None):
        if "甲" in q:
            raise RuntimeError("embed down")
        return [0.1] * 512

    monkeypatch.setattr("app.core.cloud.describe_image", lambda c, m, blob, q: "旧图描述X")
    monkeypatch.setattr(llm, "embed", fake_embed)

    stats = backfill(Settings(data_dir=tmp_path, vlm_model="qwen3-vl-plus"), session=session)

    assert stats == {"captioned": 1, "failed": 1, "remaining": 1}
    session.expire_all()
    assert session.get(Image, "imgA").caption is None  # 失败图的部分写入已回滚
    assert session.get(Image, "imgB").caption == "旧图描述X"
    assert session.get(Document, doc.id).status == "done"
    image_chunks = session.scalars(
        select(Chunk).where(Chunk.document_id == doc.id, Chunk.heading_path == "图片")
    ).all()
    assert len(image_chunks) == 1  # 只留下成功那张的 caption chunk


def test_backfill_respects_per_doc_cap_and_resumes_next_run(tmp_path, monkeypatch):
    """每轮每文档最多补 MAX_CAPTIONS_PER_DOC 张：超出留 NULL，下一次运行继续补。"""
    from scripts.caption_backfill import backfill
    from app.api.files import MAX_CAPTIONS_PER_DOC
    from app.memory import documents
    from app.models.db import Document, Image

    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    init_db(engine, dim=512)
    session = Session(engine)
    doc = documents.insert_document(session, kind="word", status="done",
                                    title="旧文档", file_path="files/old.docx", allow_cloud=1)
    session.flush()
    img_dir = tmp_path / "images"; img_dir.mkdir()
    (img_dir / "old.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
    for n in range(MAX_CAPTIONS_PER_DOC + 1):
        session.add(Image(id=f"img{n:02d}", document_id=doc.id, file_path="images/old.png"))
    session.commit()

    monkeypatch.setattr("app.core.cloud.describe_image", lambda c, m, blob, q: "旧图描述X")
    monkeypatch.setattr(llm, "embed", lambda q, model=None: [0.1] * 512)
    settings = Settings(data_dir=tmp_path, vlm_model="qwen3-vl-plus")

    first = backfill(settings, session=session)
    assert first == {"captioned": MAX_CAPTIONS_PER_DOC, "failed": 0, "remaining": 1}

    second = backfill(settings, session=session)
    assert second == {"captioned": 1, "failed": 0, "remaining": 0}
