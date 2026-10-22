import json

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.config import Settings
from app.core import llm
from app.models.db import init_db


def test_structify_tables_extracts_and_summarizes_long_block():
    from app.core.ingest.markdown import parse_markdown
    from app.core.ingest.pipeline import structify_tables, TABLE_SUMMARY_THRESHOLD
    md = ("# 成绩\n\n| 课程 | 学分 |\n|---|---|\n| 体育 | 1.0 |\n\n"
          "| 大表 | 值 |\n|---|---|\n" + "\n".join(f"| 行{i} | {i} |" for i in range(300)))
    parsed = parse_markdown(md, kind="pdf", engine_used="t")
    tables = structify_tables(parsed)
    assert len(tables) == 2 and tables[0].headers == ["课程", "学分"]
    big = [b for b in parsed.blocks if b.type == "table"][1]
    assert len(big.text) < TABLE_SUMMARY_THRESHOLD and big.text.startswith("表格（300行×2列：")
    small = [b for b in parsed.blocks if b.type == "table"][0]
    assert small.text.startswith("表格（1行×2列：")  # 短表块文本同样替换为摘要（统一规则）


def test_is_table_only_text_detects_table_doc():
    """正文只剩表格摘要行 → 表格类文档：云总结只会收到退化输入，调用方跳过不调。"""
    from app.core.ingest.pipeline import is_table_only_text

    assert is_table_only_text("表格（35行×4列：姓名、语文、数学、英语）")
    assert is_table_only_text("表格（1行×2列：A、B）\n\n表格（2行×2列：C、D）")
    assert not is_table_only_text("表格（1行×2列：A、B）\n备注：本表由教务处提供")
    assert not is_table_only_text("普通正文")
    assert not is_table_only_text("")  # 空文本不算「只剩表格」（扫描件路径另有无正文兜底）


def test_ingest_result_carries_tables(tmp_path):
    # 通过 ingest 的文件级路径验证（.md 文本文件不产 table 块 → tables 为空列表）
    from app.core.ingest.pipeline import ingest
    p = tmp_path / "a.txt"; p.write_text("普通文本", encoding="utf-8")
    result = ingest(p, ocr_name="off")
    assert result.tables == []


def test_long_table_summary_is_fts_searchable(tmp_path):
    """M8b：长表块文本替换为摘要后，摘要（含表头词）仍能被 FTS 检索命中。"""
    from app.core.ingest.chunking import chunk_document
    from app.core.ingest.markdown import parse_markdown
    from app.core.ingest.pipeline import structify_tables
    from app.memory import chunks as chunk_store, documents

    md = ("# 成绩\n\n| 课程名称 | 学分 |\n|---|---|\n"
          + "\n".join(f"| 课程{i} | 0.5 |" for i in range(300)))
    parsed = parse_markdown(md, kind="pdf", engine_used="t")
    structify_tables(parsed)
    text_chunks = chunk_document(parsed, strategy="structural")
    assert any("表格（300行×2列：课程名称" in c.text for c in text_chunks)  # 长表已替换为摘要

    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    init_db(engine, dim=512)
    session = Session(engine)
    doc = documents.insert_document(session, kind="pdf", status="done", title="成绩总表")
    session.flush()
    for i, c in enumerate(text_chunks):
        chunk = documents.insert_chunk(
            session, document_id=doc.id, seq=i, text=c.text, parent_text=c.parent_text
        )
        session.flush()
        chunk_store.index_chunk(session, chunk.id, [0.1] * 512, c.text)
    session.commit()

    assert chunk_store.fts_search(session, "课程名称", k=5)  # 表头词（≥3 字）可命中摘要块


def _fake_ingest_result_with_tables(suspect=False):
    from app.core.ingest.models import ParsedDocument, TextChunk
    from app.core.ingest.pipeline import IngestResult
    from app.core.tables import ParsedTable
    parsed = ParsedDocument(
        kind="word", title="t", full_text="正文", headings=[], blocks=[],
        engine_used="fake", ocr_used=False, needs_ocr=None,
    )
    chunks = [TextChunk(text="正文块", heading_path=None, page=None, parent_text="正文块", seq=0)]
    tables = [ParsedTable(headers=["课程", "学分"], rows=[["体育", "1.0"], ["就业指导", "0.5"]],
                          heading="成绩", suspect=suspect)]
    return IngestResult(parsed=parsed, chunks=chunks, tables=tables)


def test_process_document_writes_doc_tables(tmp_path, monkeypatch):
    """入库写入：process_document 把 IngestResult.tables 逐表写进 doc_tables（JSON 字符串）。"""
    from app.api import files
    from app.memory import documents
    from app.models.db import DocTable

    monkeypatch.setattr(files.pipeline, "ingest",
                        lambda *a, **k: _fake_ingest_result_with_tables())
    monkeypatch.setattr(llm, "embed", lambda q, model=None: [0.1] * 1024)
    settings = Settings(data_dir=tmp_path)
    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    init_db(engine, dim=1024)  # 维度须与会话工厂一致（bge-m3 → 1024）
    session = Session(engine)
    doc = documents.insert_document(session, kind="word", title="成绩单",
                                    file_path="files/x.docx", status="processing")
    session.commit()
    doc_id = doc.id
    session.close()

    files.process_document(settings, doc_id, "files/x.docx", "off", 0, 0)

    session = Session(engine)
    row = session.scalar(select(DocTable).where(DocTable.document_id == doc_id))
    assert row is not None and row.heading == "成绩"
    assert json.loads(row.headers) == ["课程", "学分"]
    assert json.loads(row.rows) == [["体育", "1.0"], ["就业指导", "0.5"]]


def test_init_db_adds_suspect_column_to_existing_doc_tables(tmp_path):
    from app.models.db import init_db
    from sqlalchemy import create_engine, text
    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE doc_tables (id TEXT PRIMARY KEY, document_id TEXT NOT NULL, "
            "heading TEXT, headers TEXT NOT NULL, rows TEXT NOT NULL, created_at TEXT NOT NULL)"
        ))
    engine.dispose()          # sqlite-vec 钩子只对新连接生效（照 test_image_caption 先例）
    init_db(engine, dim=512)
    with engine.begin() as conn:
        cols = {r[1] for r in conn.execute(text("PRAGMA table_info(doc_tables)"))}
    assert "suspect" in cols
    init_db(engine, dim=512)  # 幂等


def test_process_document_persists_suspect_flag(tmp_path, monkeypatch):
    """入库透传：ParsedTable.suspect=True 落库为 doc_tables.suspect=1。"""
    from app.api import files
    from app.memory import documents
    from app.models.db import DocTable

    monkeypatch.setattr(files.pipeline, "ingest",
                        lambda *a, **k: _fake_ingest_result_with_tables(suspect=True))
    monkeypatch.setattr(llm, "embed", lambda q, model=None: [0.1] * 1024)
    settings = Settings(data_dir=tmp_path)
    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    init_db(engine, dim=1024)
    session = Session(engine)
    doc = documents.insert_document(session, kind="word", title="成绩单",
                                    file_path="files/x.docx", status="processing")
    session.commit()
    doc_id = doc.id
    session.close()

    files.process_document(settings, doc_id, "files/x.docx", "off", 0, 0)

    session = Session(engine)
    row = session.scalar(select(DocTable).where(DocTable.document_id == doc_id))
    assert row is not None and row.suspect == 1
