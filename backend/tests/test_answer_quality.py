"""Regression for receipt lookup, poisoned cache and incomplete paper summary."""
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, text

from app import deps
from app.config import Settings
from app.core import llm, summarize
from app.main import create_app
from app.memory import documents, qa_history
from app.models.db import QaHistory, Document


@pytest.fixture
def quality(tmp_path, monkeypatch):
    settings = Settings(_env_file=None, data_dir=tmp_path, embedding_model="BAAI/bge-small-zh-v1.5",
                        rerank_engine="off", answer_cache="on", warmup=False)
    app = create_app()
    app.dependency_overrides[deps.get_settings] = lambda: settings
    monkeypatch.setattr(llm, "embed", lambda *args, **kwargs: [1.] + [0.] * 511)
    yield TestClient(app), deps._session_factory(settings), settings


def seed_doc(factory, settings, *, title="uuid-receipt.png", body="样例超市（0000样例店）重打小票，成交金额。", kind="photo"):
    path = settings.data_dir / "files" / title
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"local original")
    with factory() as s:
        doc = documents.insert_document(s, title=title, kind=kind, status="done", allow_cloud=1,
                                        parsed_text=body, file_path=f"files/{title}")
        s.commit()
        return doc.id


def test_original_receipt_found_without_cloud_and_not_poisoned_by_cached_failure(quality, monkeypatch):
    c, factory, settings = quality
    doc_id = seed_doc(factory, settings)
    seed_doc(factory, settings, title="rag.pdf", body="超市推荐系统与RAG研究", kind="document")
    with factory() as s:
        qa_history.record(s, question="帮我查找样例超市小票", answer="（回答合成失败，以下为检索到的相关记录）",
                          sources=[], documents=[], question_vec=[1.] + [0.] * 511)
        s.commit()
    monkeypatch.setattr(llm, "build_client", lambda *_: pytest.fail("Finding a known original must not call a cloud model"))
    for _ in range(2):
        result = c.post("/api/query", json={"question": "帮我查找样例超市小票"}).json()
        assert result["status"] == "ok" and result["from_cache"] is False
        assert [f["document_id"] for f in result["execution"]["files"]] == [doc_id]
        assert result["execution"]["files"][0]["available"]
        assert "合成失败" not in result["answer"]
    with factory() as s:
        assert len(qa_history.list_rows(s)) == 3  # Failed history is retained, not deleted.


def test_failed_legacy_and_structured_answers_are_never_lookup_candidates(quality):
    _, factory, _ = quality
    with factory() as s:
        good = qa_history.record(s, question="q", answer="原件在书房", sources=[], documents=[], question_vec=[1.] + [0.] * 511)
        for answer, meta in [("（回答合成失败，以下为检索到的相关记录）", {}),
                             ("部分结果", {"status": "degraded"}),
                             ("未完成", {"reading": {"complete": False}})]:
            qa_history.record(s, question="q", answer=answer, execution=meta, sources=[], documents=[], question_vec=[1.] + [0.] * 511)
        s.commit()
        hit = qa_history.lookup(s, [1.] + [0.] * 511, threshold=.98, top_k=200, ttl_hours=24)
        assert hit.id == good.id


def test_full_summary_covers_all_characters_and_does_not_mix_other_paper(quality, monkeypatch):
    c, factory, settings = quality
    body = "摘要与引言\n" + "正文不同段落。" * 1500 + "尾部结论不能丢失"
    doc_id = seed_doc(factory, settings, title="目标论文.pdf", body=body, kind="document")
    seed_doc(factory, settings, title="另一篇论文.pdf", body="不应混入的另外观点", kind="document")
    calls = []
    def chat(client, model, system, user):
        calls.append(user)
        return "# 论文总结\n\n**方法与结论**均来自目标文档。"
    monkeypatch.setattr(summarize, "_chat", chat)
    # Seed a historical partial summary with an otherwise perfect vector match.
    with factory() as s:
        qa_history.record(s, question="总结目标论文.pdf", answer="旧片段摘要", sources=[], documents=[], question_vec=[1.] + [0.] * 511)
        s.commit()
    result = c.post("/api/query", json={"question": "总结目标论文.pdf"}).json()
    assert result["status"] == "ok" and not result["from_cache"]
    source_prompts = [p for p in calls if "原文：\n" in p]
    actual = "".join(p.split("原文：\n", 1)[1] for p in source_prompts)
    assert actual == body and actual.endswith("尾部结论不能丢失")
    assert all(len(p.split("原文：\n", 1)[1]) <= 8000 for p in source_prompts)
    reading = result["execution"]["reading"]
    assert reading["complete"] and reading["covered_chars"] == reading["total_chars"] == len(body)
    assert reading["segments_completed"] == reading["segments_total"] == len(source_prompts)
    assert result["execution"]["files"][0]["document_id"] == doc_id
    before = len(calls)
    cached = c.post("/api/query", json={"question": "总结目标论文.pdf"}).json()
    assert cached["from_cache"] and len(calls) == before
    assert cached["execution"]["reading"] == reading
    history = c.get("/api/qa-history").json()["items"][0]
    detail = c.get(f'/api/workspace/history/{history["id"]}').json()
    assert detail["execution"]["reading"]["complete"]


def test_failed_segment_is_visible_and_never_cached(quality, monkeypatch):
    c, factory, settings = quality
    seed_doc(factory, settings, title="长论文.pdf", body="甲" * 8000 + "第二段必须保留" * 1100, kind="document")
    calls = []
    def chat(client, model, system, user):
        calls.append(user)
        if "第1/" in user:
            raise RuntimeError("transient network failure")
        return "第二段内容总结"
    monkeypatch.setattr(summarize, "_chat", chat)
    for _ in range(2):
        result = c.post("/api/query", json={"question": "总结长论文.pdf"}).json()
        assert result["status"] == "partial" and not result["from_cache"]
        report = result["execution"]["reading"]
        assert not report["complete"] and report["covered_chars"] < report["total_chars"]
        assert report["segments"][0]["completed"] is False
    assert len([p for p in calls if "原文：\n" in p]) >= 4


def test_unallowed_text_still_masked_in_full_summary(quality, monkeypatch):
    c, factory, settings = quality
    doc_id = seed_doc(factory, settings, title="私密论文.pdf", body="联系方式13900000003，结论仍可总结。", kind="document")
    with factory() as s:
        s.get(Document, doc_id).allow_cloud = 0; s.commit()
    prompts = []
    monkeypatch.setattr(summarize, "_chat", lambda client, model, system, user: prompts.append(user) or "已打码的论文摘要")
    result = c.post("/api/query", json={"question": "总结私密论文.pdf"}).json()
    assert result["status"] == "ok"
    assert all("13900000003" not in p for p in prompts)


def test_empty_and_truncated_model_responses_not_complete(quality, monkeypatch):
    _, factory, settings = quality
    doc_id = seed_doc(factory, settings, title="test.pdf", body="正文", kind="document")
    monkeypatch.setattr(summarize, "_chat", lambda *args: "")
    with factory() as s:
        result = summarize.summarize_with_coverage(s, doc_id, client=None, model="m")
        assert result.status == "degraded" and result.reading["covered_chars"] == 0
    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw:
        SimpleNamespace(choices=[SimpleNamespace(finish_reason="length", message=SimpleNamespace(content="截断结果"))]))))
    # Test the real _chat independently of the monkeypatch.
    monkeypatch.undo()
    with pytest.raises(RuntimeError, match="truncated"):
        summarize._chat(fake, "m", "system", "source")


def test_deleted_original_not_returned_by_file_lookup(quality):
    c, factory, settings = quality
    doc_id = seed_doc(factory, settings)
    c.delete(f"/api/documents/{doc_id}")
    with factory() as s:
        assert documents.find_files(s, "样例超市小票") == []


def test_full_summary_cache_cannot_substitute_a_different_paper(quality, monkeypatch):
    c, factory, settings = quality
    a = seed_doc(factory, settings, title="论文甲.pdf", body="甲的独立结论", kind="document")
    b = seed_doc(factory, settings, title="论文乙.pdf", body="乙的完全不同结论", kind="document")
    calls = []
    monkeypatch.setattr(summarize, "_chat", lambda client, model, system, user: calls.append(user) or "本篇论文总结")
    first = c.post("/api/query", json={"question": "总结论文甲.pdf"}).json()
    second = c.post("/api/query", json={"question": "总结论文乙.pdf"}).json()
    # The fixture deliberately gives all questions identical vectors (cosine=1).
    assert not second["from_cache"]
    assert first["execution"]["reading"]["document_id"] == a
    assert second["execution"]["reading"]["document_id"] == b
    assert len(calls) == 2
