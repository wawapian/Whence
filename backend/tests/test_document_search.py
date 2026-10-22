"""POST /api/documents/search 文档检索接口测试。"""
from fastapi.testclient import TestClient

from app import deps
from app.config import Settings
from app.core import llm
from app.main import create_app
from app.memory import chunks, documents


def _client(tmp_path, monkeypatch):
    app = create_app()
    settings = Settings(
        data_dir=tmp_path,
        embedding_model="BAAI/bge-small-zh-v1.5",
        rerank_engine="off",
    )
    app.dependency_overrides[deps.get_settings] = lambda: settings
    monkeypatch.setattr(llm, "embed", lambda q, model=None: [0.1] * 512)
    return TestClient(app), settings


def test_search_empty_returns_empty(tmp_path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch)
    resp = client.post("/api/documents/search", json={"question": "随便"})
    assert resp.status_code == 200
    assert resp.json() == {"results": []}


def test_search_returns_fragments(tmp_path, monkeypatch):
    client, settings = _client(tmp_path, monkeypatch)
    session = deps._session_factory(settings)()
    doc = documents.insert_document(session, kind="word", title="论文", status="done")
    session.flush()
    c = documents.insert_chunk(
        session,
        document_id=doc.id,
        seq=0,
        text="学习率 0.001",
        heading_path="第三章 实验",
        page=3,
        parent_text="第三组实验设置",
    )
    session.flush()
    chunks.index_chunk(session, c.id, [0.1] * 512, "学习率 0.001")
    session.commit()
    chunk_id, doc_id = c.id, doc.id
    session.close()

    resp = client.post("/api/documents/search", json={"question": "学习率"})
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) == 1
    r = results[0]
    assert r["chunk_id"] == chunk_id
    assert r["document_id"] == doc_id
    assert r["title"] == "论文"
    assert r["heading_path"] == "第三章 实验"
    assert r["page"] == 3
    assert r["text"] == "学习率 0.001"
    assert r["parent_text"] == "第三组实验设置"
