"""端到端集成：合成表格 PDF → `ingest`（真实解析链路）→ 入库 → 检索命中 + 正文剔除。

串起 Task 1–8 的接缝：坐标法出表 → 表格行组块进 FTS 索引 → doc_tables 落库 → 正文侧
不含表格区域原文。样本现造（tmp_path），不联网（`ocr_name="off"` 不加载 OCR 引擎）。
"""
import json

import pymupdf
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from helpers_pdf import make_table_pdf

# 正文用 ASCII：PDF 默认字体渲染不了 CJK（同 helpers_pdf 注释）
BODY_TEXT = "semester notes about physics lab"


def _table_pdf_with_body_page(tmp_path):
    """第 1 页表格（course/credit × PE/1.0）+ 第 2 页纯正文，用于验证「只剔除表格区域」。"""
    table_pdf = tmp_path / "grades.pdf"
    make_table_pdf(table_pdf)
    doc = pymupdf.open(str(table_pdf))
    doc.new_page().insert_text((72, 72), BODY_TEXT)
    out = tmp_path / "grades_with_body.pdf"
    doc.save(str(out))
    doc.close()
    return out


def test_ingest_table_pdf_to_search_end_to_end(tmp_path):
    from app.core.ingest.pipeline import ingest
    from app.memory import chunks as chunk_store, documents
    from app.models.db import Chunk, DocTable, init_db

    result = ingest(_table_pdf_with_body_page(tmp_path), ocr_name="off")

    # ① 坐标法出表：字段与行数据正确、无 suspect
    assert len(result.tables) == 1
    table = result.tables[0]
    assert table.headers == ["course", "credit"]
    assert table.rows == [["PE", "1.0"]]
    assert table.suspect is False

    # 入库（建库骨架同 test_long_table_summary_is_fts_searchable）
    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    init_db(engine, dim=512)
    session = Session(engine)
    doc = documents.insert_document(session, kind="pdf", status="done", title="grades")
    session.flush()
    for i, c in enumerate(result.chunks):
        chunk = documents.insert_chunk(
            session, document_id=doc.id, seq=i, text=c.text, parent_text=c.parent_text
        )
        session.flush()
        chunk_store.index_chunk(session, chunk.id, [0.1] * 512, c.text)
    documents.insert_table(
        session,
        document_id=doc.id,
        heading=table.heading,
        headers=json.dumps(table.headers),
        rows=json.dumps(table.rows),
        suspect=table.suspect,
    )
    session.commit()

    # ① doc_tables 落库正确（含 suspect=0）
    row = session.scalar(select(DocTable).where(DocTable.document_id == doc.id))
    assert row is not None
    assert json.loads(row.headers) == ["course", "credit"]
    assert json.loads(row.rows) == [["PE", "1.0"]]
    assert row.suspect == 0

    # ② FTS 命中表格行组块（表头行开头），而非摘要或正文块
    hits = chunk_store.fts_search(session, "credit", k=5)
    assert hits
    hit_texts = [session.get(Chunk, chunk_id).text for chunk_id in hits]
    assert any(t.startswith("表头：course | credit") for t in hit_texts)
    assert all(t.startswith("表头：") for t in hit_texts)  # 正文块不含 "credit"

    # ③ 正文侧不含表格区域原文，正文本身保留
    assert BODY_TEXT in result.parsed.full_text
    for cell in ("course", "credit", "PE", "1.0"):
        assert cell not in result.parsed.full_text
