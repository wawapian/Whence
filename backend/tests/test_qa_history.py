"""提问历史（qa_history 表 + 原语）与记忆库版本号。

T1：表与原语（record / list_rows / clear_all / bump / current_version / pack·unpack_vec）。
T2：全部记忆库写入路径逐条 bump + 反向断言（提问与落历史行不 bump）。
T3：POST /api/query 每次提问落一行（失败仅告警不阻断）。
T4：GET/DELETE /api/qa-history 端点。
T5：答案缓存（三态开关 + 向量/版本/TTL 命中判定 + from_cache）。
T6：消解（history 入参 + 保守回退 + resolved_question 贯通）。
"""
import json
import logging
import re
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, update
from sqlalchemy.orm import Session

from app import deps
from app.config import Settings
from app.core import answer, llm, trash
from app.main import create_app
from app.memory import chunks, documents, materials, qa_history, store
from app.models.db import QaHistory, init_db
from app.models.schemas import DocumentSource, ExtractedFact, QueryResponse, Source

MODEL = "BAAI/bge-small-zh-v1.5"


def _session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    init_db(engine, dim=512)
    return Session(engine)


# ---------- T1：表与原语 ----------


def test_record_roundtrip(tmp_path):
    """写一行 → list_rows 取回：逐字段一致（含 JSON 与向量往返）。"""
    s = _session(tmp_path)
    vec = [0.1, -0.25, 0.5, 1.0]
    row = qa_history.record(
        s,
        question="学习率是多少",
        answer="0.001",
        sources=[{"fact_id": "f1", "summary": "学习率"}],
        documents=[{"chunk_id": "c1", "document_id": "d1", "text": "片段"}],
        question_vec=vec,
    )
    s.commit()

    (got,) = qa_history.list_rows(s)
    assert got.id == row.id
    assert re.fullmatch(r"[0-9a-f]{32}", got.id)  # uuid4 hex，同 store.new_id 风格
    assert got.created_at is not None
    assert got.question == "学习率是多少"
    assert got.resolved_question is None
    assert got.answer == "0.001"
    assert json.loads(got.sources) == [{"fact_id": "f1", "summary": "学习率"}]
    assert json.loads(got.documents) == [{"chunk_id": "c1", "document_id": "d1", "text": "片段"}]
    assert got.store_version == 0
    assert got.from_cache == 0
    np.testing.assert_allclose(qa_history.unpack_vec(got.question_vec), vec, rtol=1e-6)


def test_record_resolved_question_and_from_cache(tmp_path):
    """resolved_question / from_cache 往返（缓存命中行与无 history 行都靠这两列区分）。"""
    s = _session(tmp_path)
    qa_history.record(
        s, question="那它呢", answer="答", sources=[], documents=[],
        question_vec=[0.0] * 4, resolved_question="它是指学习率吗", from_cache=True,
    )
    s.commit()

    (got,) = qa_history.list_rows(s)
    assert got.resolved_question == "它是指学习率吗"
    assert got.from_cache == 1
    assert json.loads(got.sources) == []
    assert json.loads(got.documents) == []


def test_list_rows_newest_first_with_limit_offset(tmp_path):
    s = _session(tmp_path)
    for i in range(3):
        qa_history.record(
            s, question=f"问题{i}", answer="答", sources=[], documents=[],
            question_vec=[0.0] * 4,
        )
        s.commit()

    assert [r.question for r in qa_history.list_rows(s)] == ["问题2", "问题1", "问题0"]
    assert [r.question for r in qa_history.list_rows(s, limit=2)] == ["问题2", "问题1"]
    assert [r.question for r in qa_history.list_rows(s, limit=2, offset=2)] == ["问题0"]


def test_clear_all_returns_count_and_empties(tmp_path):
    s = _session(tmp_path)
    for i in range(2):
        qa_history.record(
            s, question=f"q{i}", answer="答", sources=[], documents=[],
            question_vec=[0.0] * 4,
        )
    s.commit()

    assert qa_history.clear_all(s) == 2
    s.commit()
    assert qa_history.list_rows(s) == []


def test_current_version_starts_at_zero_then_bumps(tmp_path):
    s = _session(tmp_path)
    assert qa_history.current_version(s) == 0  # 初值 0（无 store_meta 行）
    qa_history.bump(s)
    assert qa_history.current_version(s) == 1
    qa_history.bump(s)
    assert qa_history.current_version(s) == 2


def test_version_persists_across_sessions(tmp_path):
    s = _session(tmp_path)
    qa_history.bump(s)
    s.commit()

    assert qa_history.current_version(_session(tmp_path)) == 1


def test_record_stamps_current_version(tmp_path):
    """落行时取当前版本号（缓存有效性判定 = 行版本 == 当前版本）。"""
    s = _session(tmp_path)
    qa_history.bump(s)
    qa_history.bump(s)
    qa_history.record(
        s, question="q", answer="a", sources=[], documents=[], question_vec=[0.0] * 4,
    )
    s.commit()

    assert qa_history.list_rows(s)[0].store_version == 2


def test_pack_vec_float32_roundtrip():
    blob = qa_history.pack_vec([0.1, -0.5, 1.0])
    assert isinstance(blob, bytes) and len(blob) == 12  # float32 × 3
    arr = qa_history.unpack_vec(blob)
    assert arr.dtype == np.float32
    np.testing.assert_allclose(arr, [0.1, -0.5, 1.0], rtol=1e-6)


# ---------- T2：全部写入路径 bump ----------

# grep 复核（`def insert_|def soft_delete_|def restore_|def purge_|def empty`）清单；
# `core.trash.empty_trash` / `sweep_expired` / `api/trash.py` 均只经 purge_* 落地，不单独埋点。
WRITE_PATHS = [
    "store.insert_raw",
    "store.insert_fact",
    "documents.insert_document",
    "documents.insert_chunk",
    "documents.insert_image",
    "documents.insert_table",
    "trash.soft_delete_document",
    "trash.soft_delete_record",
    "trash.restore_document",
    "trash.restore_record",
    "trash.purge_document",
    "trash.purge_record",
]


def _seed_record(s):
    """一条最小记录（已 flush，可被外键引用）。"""
    raw = store.insert_raw(s, kind="text", content="前置记录")
    s.flush()
    return raw


def _seed_document(s):
    """一份最小文档（已 flush，可被外键引用）。"""
    doc = documents.insert_document(s, kind="word", title="前置文档", status="done")
    s.flush()
    return doc


def _write_action(s, path):
    """备好前置对象，返回「恰好触发 `path` 这一条写入路径」的动作。

    前置动作自己也会 bump（insert_* 已埋点）——故测试在动作执行前才读当前版本。
    """
    if path == "store.insert_raw":
        return lambda: store.insert_raw(s, kind="text", content="新记录")
    if path == "store.insert_fact":
        raw = _seed_record(s)
        fact = ExtractedFact(type="other", subject_key="k", summary="新事实", confidence=0.9)
        return lambda: store.insert_fact(s, fact, raw.id)
    if path == "documents.insert_document":
        return lambda: documents.insert_document(s, kind="word", title="新文档", status="done")
    if path == "documents.insert_chunk":
        doc = _seed_document(s)
        return lambda: documents.insert_chunk(
            s, document_id=doc.id, seq=0, text="片段", parent_text="片段"
        )
    if path == "documents.insert_image":
        doc = _seed_document(s)
        return lambda: documents.insert_image(s, document_id=doc.id, file_path="files/x.png")
    if path == "documents.insert_table":
        doc = _seed_document(s)
        return lambda: documents.insert_table(s, document_id=doc.id, headers="[]", rows="[]")
    if path == "trash.soft_delete_document":
        doc = _seed_document(s)
        return lambda: trash.soft_delete_document(s, doc.id)
    if path == "trash.soft_delete_record":
        raw = _seed_record(s)
        return lambda: trash.soft_delete_record(s, raw.id)
    if path == "trash.restore_document":
        doc = _seed_document(s)
        trash.soft_delete_document(s, doc.id)
        return lambda: trash.restore_document(s, doc.id)
    if path == "trash.restore_record":
        raw = _seed_record(s)
        trash.soft_delete_record(s, raw.id)
        return lambda: trash.restore_record(s, raw.id)
    if path == "trash.purge_document":
        doc = _seed_document(s)
        return lambda: trash.purge_document(s, doc.id)
    if path == "trash.purge_record":
        raw = _seed_record(s)
        return lambda: trash.purge_record(s, raw.id)
    raise AssertionError(f"清单外未盘点的写入路径：{path}")


@pytest.mark.parametrize("path", WRITE_PATHS)
def test_write_paths_bump_store_version(tmp_path, path):
    s = _session(tmp_path)
    action = _write_action(s, path)
    s.commit()
    before = qa_history.current_version(s)

    action()
    s.commit()

    assert qa_history.current_version(s) == before + 1


def test_upsert_for_document_bumps_store_version(tmp_path, monkeypatch):
    """文档材料在**更晚的事务**里才落库（api/files.py：分块/表格落库、云总结之后）。

    不 bump 时，云调用窗口内提问的缓存行会以「已是最新版本」存活——把「没找到」缓存住。
    （record 侧材料与 insert_fact 同事务、已被覆盖，故不重复 bump。）
    """
    monkeypatch.setattr(llm, "embed", lambda text, model=None: [0.1] * 512)
    s = _session(tmp_path)
    doc = _seed_document(s)
    s.commit()
    before = qa_history.current_version(s)

    materials.upsert_for_document(s, doc, model=MODEL)
    s.commit()

    assert qa_history.current_version(s) == before + 1


def test_trash_failed_ops_do_not_bump(tmp_path):
    """bump 只在成功点：目标不存在一律 False（无写入），不得动版本号。"""
    s = _session(tmp_path)

    assert trash.soft_delete_document(s, "nope") is False
    assert trash.soft_delete_record(s, "nope") is False
    assert trash.restore_document(s, "nope") is False
    assert trash.restore_record(s, "nope") is False
    assert trash.purge_document(s, "nope") is False
    assert trash.purge_record(s, "nope") is False

    s.commit()
    assert qa_history.current_version(s) == 0


def test_restore_live_object_does_not_bump(tmp_path):
    """恢复不在回收站的活对象 = 无写入（False），不得 bump。"""
    s = _session(tmp_path)
    doc = _seed_document(s)
    raw = _seed_record(s)
    s.commit()
    before = qa_history.current_version(s)

    assert trash.restore_document(s, doc.id) is False
    assert trash.restore_record(s, raw.id) is False
    s.commit()

    assert qa_history.current_version(s) == before


def test_empty_trash_bumps_via_purge(tmp_path):
    """empty_trash 不自行删行：逐条走 purge_*（已埋点）——传递链钉死。"""
    s = _session(tmp_path)
    raw = _seed_record(s)
    doc = _seed_document(s)
    trash.soft_delete_record(s, raw.id)
    trash.soft_delete_document(s, doc.id)
    s.commit()
    before = qa_history.current_version(s)

    counts = trash.empty_trash(s)

    assert counts == {"records": 1, "documents": 1}
    assert qa_history.current_version(s) == before + 2  # 每条 purge 各 +1（1 记录 + 1 文档）


def test_record_does_not_bump_store_version(tmp_path):
    """写历史行不是记忆库写入：不动版本号（也不建 store_meta 行）。"""
    s = _session(tmp_path)
    qa_history.record(
        s, question="q", answer="a", sources=[], documents=[], question_vec=[0.0] * 4,
    )
    s.commit()

    assert qa_history.list_rows(s)  # 行确实落了
    assert qa_history.current_version(s) == 0


def _settings(tmp_path):
    return Settings(
        _env_file=None, data_dir=tmp_path, embedding_model=MODEL, rerank_engine="off"
    )


def _direct_answer_client(content):
    """单轮直接回答的假 LLM 客户端（不调工具）。"""
    message = type("Msg", (), {"content": content, "tool_calls": None})()
    response = type("Resp", (), {"choices": [type("Choice", (), {"message": message})()]})()

    class _Completions:
        def create(self, **kwargs):
            return response

    return type("Client", (), {"chat": type("Chat", (), {"completions": _Completions()})()})()


def _seed_indexed_document(s):
    """一份可检索的文档（分块 + 向量 + 材料索引）——材料层命中才会走完整问答管线。"""
    doc = documents.insert_document(
        s, kind="word", status="done", title="实验报告", allow_cloud=1,
        parsed_text="第三组学习率 0.001",
    )
    s.flush()
    chunk = documents.insert_chunk(
        s, document_id=doc.id, seq=0, text="第三组学习率 0.001", parent_text="第三组学习率 0.001"
    )
    s.flush()
    chunks.index_chunk(s, chunk.id, [0.1] * 512, "第三组学习率 0.001")
    materials.upsert_for_document(s, doc, model=MODEL)
    return doc


def test_ask_flow_does_not_bump(tmp_path, monkeypatch):
    """提问流程（检索 + 回答，即 /api/query 的问答入口）全程只读：不 bump。"""
    monkeypatch.setattr(llm, "embed", lambda text, model=None: [0.1] * 512)
    s = _session(tmp_path)
    _seed_indexed_document(s)  # 内含材料 upsert（embed 替身），不加载真模型
    s.commit()
    monkeypatch.setattr(
        llm, "build_client", lambda settings: _direct_answer_client("学习率是 0.001")
    )
    before = qa_history.current_version(s)

    resp = answer.answer(s, "学习率多少", _settings(tmp_path))

    assert resp.answer == "学习率是 0.001"
    assert qa_history.current_version(s) == before


# ---------- T3：每次提问落行（失败不阻断） ----------


def _client(tmp_path, **overrides):
    """TestClient + 依赖覆盖到 tmp 库；勿用 `with`——lifespan 会打真 data_dir。"""
    app = create_app()
    app.dependency_overrides[deps.get_settings] = lambda: Settings(
        _env_file=None, data_dir=tmp_path, embedding_model=MODEL, rerank_engine="off",
        **overrides,
    )
    return TestClient(app)


def test_query_endpoint_records_history_row(tmp_path, monkeypatch):
    """POST /api/query 每次调用恒定落一行：question/answer/sources/documents 逐字一致；
    本期口径 from_cache=0、resolved_question=None、向量=原问题。"""
    monkeypatch.setattr(llm, "embed", lambda text, model=None: [0.5] * 512)
    fixed = QueryResponse(
        answer="学习率是 0.001",
        sources=[Source(fact_id="f1", summary="学习率 0.001")],
        documents=[DocumentSource(chunk_id="c1", document_id="d1", text="第三组学习率 0.001")],
    )
    monkeypatch.setattr(answer, "answer", lambda session, question, settings: fixed)

    resp = _client(tmp_path).post("/api/query", json={"question": "学习率多少"})

    assert resp.status_code == 200
    assert resp.json()["answer"] == "学习率是 0.001"
    rows = qa_history.list_rows(_session(tmp_path))
    assert len(rows) == 1
    row = rows[0]
    assert row.question == "学习率多少"
    assert row.answer == "学习率是 0.001"
    assert row.resolved_question is None
    assert row.from_cache == 0
    # sources/documents 存 model_dump() 后的 dict 列表（pydantic 对象不可 JSON 序列化）
    assert json.loads(row.sources) == [s.model_dump() for s in fixed.sources]
    assert json.loads(row.documents) == [d.model_dump() for d in fixed.documents]
    np.testing.assert_allclose(qa_history.unpack_vec(row.question_vec), [0.5] * 512, rtol=1e-6)


def test_query_endpoint_survives_record_failure(tmp_path, monkeypatch, caplog):
    """落行抛异常 → 响应照常 200（红线：只 warning，不阻断回答）。"""
    monkeypatch.setattr(llm, "embed", lambda text, model=None: [0.5] * 512)
    monkeypatch.setattr(
        answer, "answer", lambda session, question, settings: QueryResponse(answer="答")
    )

    def boom(*args, **kwargs):
        raise RuntimeError("落行炸了")

    monkeypatch.setattr(qa_history, "record", boom)

    with caplog.at_level(logging.WARNING, logger="app.api.query"):
        resp = _client(tmp_path).post("/api/query", json={"question": "学习率多少"})

    assert resp.status_code == 200
    assert resp.json()["answer"] == "答"
    assert "提问历史落行失败" in caplog.text
    assert qa_history.list_rows(_session(tmp_path)) == []


# ---------- T4：读/清接口 ----------


def test_qa_history_endpoint_lists_newest_first_with_limit_and_total(tmp_path):
    """GET 倒序、limit/offset 生效、total 恒为全表条数；created_at 出口带本机偏移。"""
    s = _session(tmp_path)
    for i in range(2):
        qa_history.record(
            s, question=f"问题{i}", answer=f"答{i}", sources=[], documents=[],
            question_vec=[0.0] * 512,
        )
        s.commit()

    client = _client(tmp_path)
    body = client.get("/api/qa-history").json()

    assert [i["question"] for i in body["items"]] == ["问题1", "问题0"]
    assert body["total"] == 2
    item = body["items"][0]
    assert set(item) == {
        "id", "created_at", "question", "resolved_question", "answer", "from_cache"
    }
    assert item["answer"] == "答1"
    assert item["resolved_question"] is None
    assert item["from_cache"] is False
    # created_at：ORM 读回是 naive-UTC，出口须经 to_local（直拼 isoformat 无偏移、还差一个时区）
    got = datetime.fromisoformat(item["created_at"])
    assert got.tzinfo is not None
    assert got == s.get(QaHistory, item["id"]).created_at.replace(tzinfo=timezone.utc)

    page = client.get("/api/qa-history?limit=1").json()
    assert [i["question"] for i in page["items"]] == ["问题1"]
    assert page["total"] == 2  # total 不受 limit 影响
    assert client.get("/api/qa-history?offset=2").json()["items"] == []


def test_qa_history_endpoint_delete_clears_all(tmp_path):
    """DELETE → 204 空体；全清后 GET 空（total=0），库表确已清空。"""
    s = _session(tmp_path)
    for i in range(2):
        qa_history.record(
            s, question=f"q{i}", answer="答", sources=[], documents=[],
            question_vec=[0.0] * 512,
        )
    s.commit()
    client = _client(tmp_path)

    resp = client.delete("/api/qa-history")

    assert resp.status_code == 204
    assert resp.content == b""
    assert client.get("/api/qa-history").json() == {"items": [], "total": 0}
    assert qa_history.list_rows(s) == []


# ---------- T5：答案缓存（三态 + 向量/版本/TTL 命中判定） ----------


def _recording_pipeline(monkeypatch, responses):
    """monkeypatch 问答管线：逐字记录每次入参问题；按序出 answers（用尽后重复最后一个）。"""
    calls = []

    def fake(session, question, settings):
        calls.append(question)
        return responses[min(len(calls) - 1, len(responses) - 1)]

    monkeypatch.setattr(answer, "answer", fake)
    return calls


def test_cache_off_repeat_question_never_hits(tmp_path, monkeypatch):
    """off（默认）：不查缓存——重复问两次都走管线，两行 from_cache=0。"""
    monkeypatch.setattr(llm, "embed", lambda text, model=None: [0.5] * 512)
    calls = _recording_pipeline(
        monkeypatch, [QueryResponse(answer="一答"), QueryResponse(answer="二答")]
    )
    client = _client(tmp_path)

    first = client.post("/api/query", json={"question": "学习率多少"})
    second = client.post("/api/query", json={"question": "学习率多少"})

    assert first.json()["from_cache"] is False
    assert second.json()["from_cache"] is False
    assert second.json()["answer"] == "二答"  # 仍未命中，走管线
    assert calls == ["学习率多少", "学习率多少"]
    assert [r.from_cache for r in qa_history.list_rows(_session(tmp_path))] == [0, 0]


def test_cache_on_hit_returns_stored_payload_and_records_hit_row(tmp_path, monkeypatch):
    """on：第二问命中——answer/sources/documents 与存下的逐字一致、from_cache=true；
    命中不再调管线，并落一行 from_cache=1。"""
    monkeypatch.setattr(llm, "embed", lambda text, model=None: [0.5] * 512)
    fixed = QueryResponse(
        answer="学习率是 0.001",
        sources=[Source(fact_id="f1", summary="学习率 0.001")],
        documents=[DocumentSource(chunk_id="c1", document_id="d1", text="第三组学习率 0.001")],
    )
    calls = _recording_pipeline(monkeypatch, [fixed])
    client = _client(tmp_path, answer_cache="on")

    first = client.post("/api/query", json={"question": "学习率多少"})
    second = client.post("/api/query", json={"question": "学习率多少"})

    assert first.json()["from_cache"] is False
    body = second.json()
    assert body["from_cache"] is True
    assert body["answer"] == "学习率是 0.001"
    assert body["sources"] == [s.model_dump() for s in fixed.sources]
    assert body["documents"] == [d.model_dump() for d in fixed.documents]
    assert calls == ["学习率多少"]  # 命中时不再调管线
    rows = qa_history.list_rows(_session(tmp_path))
    assert [r.from_cache for r in rows] == [1, 0]  # 倒序：命中落行在前
    assert rows[0].answer == "学习率是 0.001"


def test_cache_similar_but_different_question_not_hit(tmp_path, monkeypatch):
    """易混不同问题（不同向量）在阈值下不命中：静默走管线（宁严勿松）。"""
    def fake_embed(text, model=None):
        v = [0.0] * 512
        v[0 if "学习率" in text else 1] = 1.0
        return v

    monkeypatch.setattr(llm, "embed", fake_embed)
    calls = _recording_pipeline(
        monkeypatch, [QueryResponse(answer="一答"), QueryResponse(answer="二答")]
    )
    client = _client(tmp_path, answer_cache="on")

    client.post("/api/query", json={"question": "学习率是多少"})
    second = client.post("/api/query", json={"question": "今天天气怎么样"})

    assert second.json()["from_cache"] is False
    assert second.json()["answer"] == "二答"  # 未命中 → 走管线
    assert calls == ["学习率是多少", "今天天气怎么样"]
    assert [r.from_cache for r in qa_history.list_rows(_session(tmp_path))] == [0, 0]


def test_cache_row_invalidated_after_store_bump(tmp_path, monkeypatch):
    """记忆库写入（版本 +1）使旧行失效：不命中，走管线。"""
    monkeypatch.setattr(llm, "embed", lambda text, model=None: [0.5] * 512)
    calls = _recording_pipeline(
        monkeypatch, [QueryResponse(answer="一答"), QueryResponse(answer="二答")]
    )
    client = _client(tmp_path, answer_cache="on")

    client.post("/api/query", json={"question": "学习率多少"})
    s = _session(tmp_path)
    qa_history.bump(s)
    s.commit()

    second = client.post("/api/query", json={"question": "学习率多少"})

    assert second.json()["from_cache"] is False
    assert second.json()["answer"] == "二答"
    assert calls == ["学习率多少", "学习率多少"]


def test_cache_row_expired_by_ttl_not_hit(tmp_path, monkeypatch):
    """TTL 外的行不参与命中（created_at 回拨 25h > 默认 24h）：走管线。"""
    monkeypatch.setattr(llm, "embed", lambda text, model=None: [0.5] * 512)
    calls = _recording_pipeline(
        monkeypatch, [QueryResponse(answer="一答"), QueryResponse(answer="二答")]
    )
    client = _client(tmp_path, answer_cache="on")

    client.post("/api/query", json={"question": "学习率多少"})
    s = _session(tmp_path)
    s.execute(update(QaHistory).values(created_at=datetime.now(timezone.utc) - timedelta(hours=25)))
    s.commit()

    second = client.post("/api/query", json={"question": "学习率多少"})

    assert second.json()["from_cache"] is False
    assert second.json()["answer"] == "二答"
    assert calls == ["学习率多少", "学习率多少"]


def test_cache_lookup_compares_only_recent_top_k_rows(tmp_path):
    """lookup 只比对最近 top_k 行：窗口外的相似行不参与命中。"""
    s = _session(tmp_path)
    qa_history.record(
        s, question="旧问题", answer="旧答", sources=[], documents=[],
        question_vec=[1.0] + [0.0] * 3,
    )
    s.commit()
    qa_history.record(
        s, question="新问题", answer="新答", sources=[], documents=[],
        question_vec=[0.0, 1.0, 0.0, 0.0],
    )
    s.commit()

    vec = [1.0] + [0.0] * 3
    assert qa_history.lookup(s, vec, threshold=0.9, top_k=1, ttl_hours=24) is None
    hit = qa_history.lookup(s, vec, threshold=0.9, top_k=2, ttl_hours=24)
    assert hit is not None and hit.question == "旧问题"


def test_cache_log_mode_identical_response_with_info_log(tmp_path, monkeypatch, caplog):
    """log：只记「若开会否命中」，响应与 off 逐字一致且照走管线；
    日志含 问题/匹配行/相似度/阈值。"""
    monkeypatch.setattr(llm, "embed", lambda text, model=None: [0.5] * 512)
    fixed = QueryResponse(answer="答", sources=[Source(fact_id="f1")])
    calls = _recording_pipeline(monkeypatch, [fixed])

    off_resp = _client(tmp_path).post("/api/query", json={"question": "学习率多少"})

    # 同库第二问：已有一行版本一致、TTL 内、同向量的候选——若开 on 会命中（log 只记不改）
    with caplog.at_level(logging.INFO, logger="app.api.query"):
        log_resp = _client(tmp_path, answer_cache="log").post(
            "/api/query", json={"question": "学习率多少"}
        )

    assert log_resp.content == off_resp.content  # 逐字一致
    assert log_resp.json()["from_cache"] is False
    assert calls == ["学习率多少", "学习率多少"]  # log 照走管线
    assert "答案缓存" in caplog.text and "阈值" in caplog.text


def test_cache_on_hit_logs_audit_info(tmp_path, monkeypatch, caplog):
    """on 命中（终审 Minor#1）：打同款 info 审计（问题/匹配行 id/相似度/阈值）——
    0.95 薄边距下留痕；命中行为不变（直答 + 不调管线）。"""
    def fake_embed(text, model=None):
        v = [0.0] * 512
        v[0] = 1.0
        return v

    monkeypatch.setattr(llm, "embed", fake_embed)
    fixed = QueryResponse(answer="学习率是 0.001", sources=[Source(fact_id="f1")])
    calls = _recording_pipeline(monkeypatch, [fixed])
    client = _client(tmp_path, answer_cache="on")

    client.post("/api/query", json={"question": "学习率多少"})
    with caplog.at_level(logging.INFO, logger="app.api.query"):
        second = client.post("/api/query", json={"question": "学习率多少"})

    assert second.json()["from_cache"] is True  # 命中直答，行为不变
    assert calls == ["学习率多少"]  # 命中不调管线
    # 倒序：命中落行在前，命中的是「第一问」那行
    first_row = qa_history.list_rows(_session(tmp_path))[1]
    assert "答案缓存(on)" in caplog.text
    assert f"匹配行={first_row.id}" in caplog.text
    assert "相似度=1.0000" in caplog.text  # 同向量余弦 = 1
    assert "阈值=0.98" in caplog.text  # 2026-10-16 由 0.95 上调（见 config 注释）


def test_cache_on_miss_does_not_log_audit_info(tmp_path, monkeypatch, caplog):
    """on 未命中：不打审计 info（防噪音——miss 仍只 log 档记）。"""
    monkeypatch.setattr(llm, "embed", lambda text, model=None: [0.5] * 512)
    _recording_pipeline(monkeypatch, [QueryResponse(answer="答")])
    client = _client(tmp_path, answer_cache="on")

    with caplog.at_level(logging.INFO, logger="app.api.query"):
        resp = client.post("/api/query", json={"question": "学习率多少"})

    assert resp.json()["from_cache"] is False
    assert "答案缓存" not in caplog.text


def test_cache_dim_mismatch_degrades_to_pipeline(tmp_path, monkeypatch, caplog):
    """旧行向量维度不符（换 embedding 模型后的真实触发面）：查缓存异常 → 200 + warning，
    按未命中走管线，不 500。"""
    monkeypatch.setattr(llm, "embed", lambda text, model=None: [0.5] * 512)
    s = _session(tmp_path)
    qa_history.record(
        s, question="旧问题", answer="旧答", sources=[], documents=[],
        question_vec=[0.1, 0.2],  # 短向量：与 512 维查询向量不符（np.dot 抛 ValueError）
    )
    s.commit()
    calls = _recording_pipeline(monkeypatch, [QueryResponse(answer="新答")])

    with caplog.at_level(logging.WARNING, logger="app.api.query"):
        resp = _client(tmp_path, answer_cache="on").post(
            "/api/query", json={"question": "学习率多少"}
        )

    assert resp.status_code == 200
    assert resp.json()["answer"] == "新答"  # 未命中 → 管线作答
    assert resp.json()["from_cache"] is False
    assert "答案缓存" in caplog.text
    assert calls == ["学习率多少"]


# ---------- T6：消解（history 入参 + 保守回退 + resolved_question 贯通） ----------


class _FakeResolverClient:
    """消解假云客户端：create 记录每次 messages；按 content/error 返回或抛出。"""

    def __init__(self, content, error, prompts):
        class _Completions:
            def create(inner, **kwargs):
                prompts.append(kwargs["messages"])
                if error is not None:
                    raise error
                message = type("Msg", (), {"content": content, "tool_calls": None})()
                return type(
                    "Resp", (), {"choices": [type("Choice", (), {"message": message})()]}
                )()

        self.chat = type("Chat", (), {"completions": _Completions()})()


def _fake_resolver(monkeypatch, content="", error=None):
    """把 llm.build_client 换成消解假客户端；返回记录到的 prompts（每次调用一条）。"""
    prompts = []
    monkeypatch.setattr(
        llm, "build_client", lambda settings: _FakeResolverClient(content, error, prompts)
    )
    return prompts


def _prompt_text(messages):
    return "\n".join(m["content"] for m in messages)


def test_history_resolves_question_before_pipeline(tmp_path, monkeypatch):
    """history 非空：消解输出进管线（原句被替换）；响应与落行带 resolved_question。"""
    monkeypatch.setattr(llm, "embed", lambda text, model=None: [0.5] * 512)
    prompts = _fake_resolver(monkeypatch, content="学习率是多少")
    calls = _recording_pipeline(monkeypatch, [QueryResponse(answer="0.001")])
    history = [{"question": "学习率多少", "answer": "0.001"}]

    resp = _client(tmp_path).post("/api/query", json={"question": "那它呢", "history": history})

    assert resp.status_code == 200
    assert calls == ["学习率是多少"]  # 管线收到独立问题
    assert resp.json()["answer"] == "0.001"
    assert resp.json()["resolved_question"] == "学习率是多少"
    assert len(prompts) == 1
    prompt = _prompt_text(prompts[0])
    assert "那它呢" in prompt and "0.001" in prompt  # 窗口（问+答）与当前句都进 prompt
    (row,) = qa_history.list_rows(_session(tmp_path))
    assert row.question == "那它呢"  # 原文照存
    assert row.resolved_question == "学习率是多少"


def test_resolve_failure_falls_back_to_original(tmp_path, monkeypatch, caplog):
    """消解云调用异常 → 原句进管线 + warning，不阻断。"""
    monkeypatch.setattr(llm, "embed", lambda text, model=None: [0.5] * 512)
    _fake_resolver(monkeypatch, error=RuntimeError("云不可用"))
    calls = _recording_pipeline(monkeypatch, [QueryResponse(answer="答")])
    history = [{"question": "学习率多少", "answer": "0.001"}]

    with caplog.at_level(logging.WARNING, logger="app.api.query"):
        resp = _client(tmp_path).post("/api/query", json={"question": "那它呢", "history": history})

    assert resp.status_code == 200
    assert calls == ["那它呢"]
    assert "指代消解失败" in caplog.text
    (row,) = qa_history.list_rows(_session(tmp_path))
    assert row.resolved_question == "那它呢"  # 实际使用值 = 原句


def test_resolve_empty_output_falls_back_to_original(tmp_path, monkeypatch, caplog):
    """消解返回空输出 → 原句进管线 + warning（保守回退另一分支）。"""
    monkeypatch.setattr(llm, "embed", lambda text, model=None: [0.5] * 512)
    _fake_resolver(monkeypatch, content="")
    calls = _recording_pipeline(monkeypatch, [QueryResponse(answer="答")])
    history = [{"question": "学习率多少", "answer": "0.001"}]

    with caplog.at_level(logging.WARNING, logger="app.api.query"):
        resp = _client(tmp_path).post("/api/query", json={"question": "那它呢", "history": history})

    assert resp.status_code == 200
    assert calls == ["那它呢"]
    assert "指代消解返回空" in caplog.text
    (row,) = qa_history.list_rows(_session(tmp_path))
    assert row.resolved_question == "那它呢"  # 实际使用值 = 原句


def test_history_window_truncates_to_last_ten(tmp_path, monkeypatch):
    """服务端截最近 qa_history_window 组：12 组入参只有后 10 组进消解 prompt。"""
    monkeypatch.setattr(llm, "embed", lambda text, model=None: [0.5] * 512)
    prompts = _fake_resolver(monkeypatch, content="独立问题")
    _recording_pipeline(monkeypatch, [QueryResponse(answer="答")])
    history = [{"question": f"问题{i}。", "answer": f"答{i}。"} for i in range(12)]

    _client(tmp_path).post("/api/query", json={"question": "那它呢", "history": history})

    assert len(prompts) == 1
    prompt = _prompt_text(prompts[0])
    assert "问题2。" in prompt and "问题11。" in prompt  # 最新 10 组在内
    assert "问题0。" not in prompt and "问题1。" not in prompt  # 最旧 2 组被截掉


@pytest.mark.parametrize("window", [0, -1])
def test_history_window_nonpositive_guard_returns_original_without_cloud(
    tmp_path, monkeypatch, window
):
    """window<=0 护栏（终审 Minor#2）：0 会让切片退化为全窗口（history[-0:]，语义反转）——
    守卫直接原句返回、零云调用（不发起消解）。"""
    monkeypatch.setattr(llm, "embed", lambda text, model=None: [0.5] * 512)
    prompts = _fake_resolver(monkeypatch, content="不该被用到")
    calls = _recording_pipeline(monkeypatch, [QueryResponse(answer="答")])
    history = [{"question": "学习率多少", "answer": "0.001"}]

    resp = _client(tmp_path, qa_history_window=window).post(
        "/api/query", json={"question": "那它呢", "history": history}
    )

    assert resp.status_code == 200
    assert calls == ["那它呢"]  # 原句进管线
    assert prompts == []  # 零云调用（消解替身未被发起）
    assert resp.json()["resolved_question"] == "那它呢"  # 实际使用值 = 原句


def test_resolve_prompt_forbids_renaming_entities(tmp_path, monkeypatch):
    """消解 prompt 硬约束钉住（发现#2-②）：只补全省略/消解指代、严禁改写改名、不回答。

    实测回归：修复前把「耳机盒子呢」消解成「外包装盒放在哪里」（替用户改了实体名）。
    """
    monkeypatch.setattr(llm, "embed", lambda text, model=None: [0.5] * 512)
    prompts = _fake_resolver(monkeypatch, content="独立问题")
    _recording_pipeline(monkeypatch, [QueryResponse(answer="答")])
    history = [{"question": "耳机盒子放哪了", "answer": "在进门左边立柜里"}]

    _client(tmp_path).post("/api/query", json={"question": "那它呢", "history": history})

    prompt = _prompt_text(prompts[0])
    assert "补全省略" in prompt and "消解指代" in prompt
    assert "改写" in prompt and "重新命名" in prompt and "原样保留" in prompt
    assert "不要回答问题" in prompt
    assert "原样返回" in prompt  # 拿不准一律原样返回（保守回退保留）


def test_no_history_keeps_resolved_question_null(tmp_path, monkeypatch):
    """无 history（或空窗）：不消解、不调云，resolved_question 恒 None。"""
    monkeypatch.setattr(llm, "embed", lambda text, model=None: [0.5] * 512)
    prompts = _fake_resolver(monkeypatch, content="不该被用到")
    _recording_pipeline(monkeypatch, [QueryResponse(answer="答")])
    client = _client(tmp_path)

    first = client.post("/api/query", json={"question": "学习率多少"})
    second = client.post("/api/query", json={"question": "学习率多少", "history": []})

    assert first.json()["resolved_question"] is None
    assert second.json()["resolved_question"] is None
    assert prompts == []  # 无窗口 → 不发起消解调用
    assert all(r.resolved_question is None for r in qa_history.list_rows(_session(tmp_path)))


def test_cache_on_key_is_resolved_question(tmp_path, monkeypatch):
    """缓存 on：先消解后查缓存——键 = 独立问题向量；带同 history 重复问直接命中，不再调管线。"""
    def fake_embed(text, model=None):
        v = [0.0] * 512
        v[0 if text == "独立问题X" else 1] = 1.0
        return v

    monkeypatch.setattr(llm, "embed", fake_embed)
    prompts = _fake_resolver(monkeypatch, content="独立问题X")
    calls = _recording_pipeline(monkeypatch, [QueryResponse(answer="答")])
    client = _client(tmp_path, answer_cache="on")
    payload = {"question": "那它呢", "history": [{"question": "学习率多少", "answer": "0.001"}]}

    first = client.post("/api/query", json=payload)
    second = client.post("/api/query", json=payload)

    assert first.json()["from_cache"] is False
    assert second.json()["from_cache"] is True
    assert second.json()["answer"] == "答"
    assert second.json()["resolved_question"] == "独立问题X"
    assert calls == ["独立问题X"]  # 仅第一问走管线
    assert len(prompts) == 2  # 消解先行：两问都先消解（第二问才以独立问题查中）
    assert [r.from_cache for r in qa_history.list_rows(_session(tmp_path))] == [1, 0]


def test_default_threshold_blocks_one_char_antonym(tmp_path):
    """默认阈值必须挡住「上个月 vs 这个月」这类**一字之差、语义相反**的问题对。

    实测（本机 bge-m3）：「我这个月一共花了多少钱？」与「上个月我一共花了多少钱？」
    余弦 **0.9749**。旧默认 0.95 会让它命中——评测语料 q-0031 就因此拿到了 q-0029 的
    答案（两答逐字相同，用户问"上个月"却得到"这个月"）。新默认 0.98 把它挡在外面，
    而**原句复问（余弦 1.0）照旧命中**（加速器价值不减）。
    """
    import math

    s = _session(tmp_path)
    cos = 0.9749
    stored = [cos, math.sqrt(1 - cos * cos)] + [0.0] * 510  # 与 [1,0,...] 的余弦恰为 0.9749
    qa_history.record(s, question="我这个月一共花了多少钱？", answer="你这个月…933 元",
                      sources=[], documents=[], question_vec=stored)
    s.commit()

    hit = qa_history.lookup(s, [1.0] + [0.0] * 511,
                            threshold=Settings(_env_file=None).answer_cache_threshold,
                            top_k=200, ttl_hours=24)

    assert hit is None
