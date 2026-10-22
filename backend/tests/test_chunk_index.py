from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from app.memory import chunks, documents
from app.models.db import Chunk, init_db


def _session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    init_db(engine, dim=4)
    return Session(engine)


def test_index_and_search_roundtrip(tmp_path):
    session = _session(tmp_path)
    doc = documents.insert_document(session, kind="word", status="done")
    session.flush()
    c = documents.insert_chunk(session, document_id=doc.id, seq=0,
                               text="学习率 0.001", parent_text="第三组实验设置")
    session.flush()
    chunks.index_chunk(session, c.id, [0.1] * 4, "学习率 0.001")
    session.commit()
    assert chunks.vector_search(session, [0.1] * 4, k=5) == [c.id]
    assert chunks.fts_search(session, "学习率", k=5) == [c.id]
    assert chunks.chunk_text(session, c.id) == "学习率 0.001"


def test_index_chunk_backfills_vec_rowid(tmp_path):
    session = _session(tmp_path)
    doc = documents.insert_document(session, kind="word", status="done")
    session.flush()
    c = documents.insert_chunk(session, document_id=doc.id, seq=0, text="x", parent_text="x")
    session.flush()
    chunks.index_chunk(session, c.id, [0.1] * 4, "x")
    session.commit()
    assert session.get(Chunk, c.id).vec_rowid is not None


def test_unindexed_chunk_not_in_vector_search(tmp_path):
    session = _session(tmp_path)
    doc = documents.insert_document(session, kind="word", status="done")
    session.flush()
    documents.insert_chunk(session, document_id=doc.id, seq=0, text="未索引", parent_text="x")
    session.commit()
    assert chunks.vector_search(session, [0.1] * 4, k=5) == []
