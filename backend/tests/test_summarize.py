from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.core import summarize
from app.memory import documents
from app.models.db import init_db


def _session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    init_db(engine, dim=4)
    return Session(engine)


def test_short_doc_one_shot(tmp_path, monkeypatch):
    session = _session(tmp_path)
    doc = documents.insert_document(session, kind="txt", status="done")
    session.flush()
    documents.insert_chunk(session, document_id=doc.id, seq=0, text="第一段", parent_text="")
    documents.insert_chunk(session, document_id=doc.id, seq=1, text="第二段", parent_text="")
    session.commit()

    calls = []

    def fake_chat(client, model, system, user):
        calls.append(user)
        return "总结"

    monkeypatch.setattr(summarize, "_chat", fake_chat)
    out = summarize.summarize_document(session, doc.id, client=None, model="m", max_chars=100000)
    assert len(calls) == 1
    assert out == "总结"
    assert "第一段" in calls[0]


def test_long_doc_map_reduce(tmp_path, monkeypatch):
    session = _session(tmp_path)
    doc = documents.insert_document(session, kind="txt", status="done")
    session.flush()
    # 2 个顶级标题，各 2 个 chunk，全文远大于 max_chars=10
    for i in range(4):
        heading = "第一章" if i < 2 else "第二章"
        documents.insert_chunk(
            session,
            document_id=doc.id,
            seq=i,
            text="很长的内容" * 10,
            heading_path=f"{heading}>小节{i}",
            parent_text="",
        )
    session.commit()

    calls = []

    def fake_chat(client, model, system, user):
        calls.append(user)
        return "小节摘要"

    monkeypatch.setattr(summarize, "_chat", fake_chat)
    out = summarize.summarize_document(session, doc.id, client=None, model="m", max_chars=10)
    # Each source request is bounded; no oversized top-level chapter bypasses the limit.
    assert len(calls) == 21
    assert all(len(prompt.split("原文：\n", 1)[1]) <= 10 for prompt in calls[:-1])
    assert out == "小节摘要"


def test_no_chunks_uses_parsed_text(tmp_path, monkeypatch):
    session = _session(tmp_path)
    doc = documents.insert_document(session, kind="txt", status="done", parsed_text="全文原文")
    session.commit()

    calls = []

    def fake_chat(client, model, system, user):
        calls.append(user)
        return "总结"

    monkeypatch.setattr(summarize, "_chat", fake_chat)
    out = summarize.summarize_document(session, doc.id, client=None, model="m")
    assert len(calls) == 1
    assert out == "总结"
    assert "全文原文" in calls[0]


def test_long_parsed_text_no_chunks_is_fully_segmented(tmp_path, monkeypatch):
    """Even without chunks, every body character is sent in bounded requests."""
    session = _session(tmp_path)
    long_text = "长" * 30000
    doc = documents.insert_document(session, kind="txt", status="done", parsed_text=long_text)
    session.commit()

    calls = []

    def fake_chat(client, model, system, user):
        calls.append(user)
        return "总结"

    monkeypatch.setattr(summarize, "_chat", fake_chat)
    out = summarize.summarize_document(session, doc.id, client=None, model="m", max_chars=20000)
    assert len(calls) == 3
    assert out == "总结"
    assert "".join(p.split("原文：\n", 1)[1] for p in calls[:-1]) == long_text


def test_llm_failure_is_explicit_not_a_fake_summary(tmp_path, monkeypatch):
    session = _session(tmp_path)
    doc = documents.insert_document(session, kind="txt", status="done")
    session.flush()
    documents.insert_chunk(session, document_id=doc.id, seq=0, text="原始内容", parent_text="")
    session.commit()

    def boom(*a, **k):
        raise Exception("llm down")

    monkeypatch.setattr(summarize, "_chat", boom)
    out = summarize.summarize_document(session, doc.id, client=None, model="m")
    assert "全文总结未完成" in out
